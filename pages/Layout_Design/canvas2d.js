    }
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
      phyRw = polylineRawDurationLinearRetSpeed(runwayPts, vTd, vRetIn);
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
    const phyRet = (retPts.length >= 2 && retPathLen > 1e-3) ? polylineRawDurationLinearRetSpeed(retPts, vRetIn, vRetOut) : 0;
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
    const vRetIn = parts.vRetIn;
    const vRetOut = parts.vRetOut;
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
        ? polylineTimelineLinearRetSpeed(runwayPts, tCur, tSegEnd, vTd, vRetIn)
        : polylineSpeedScaledToWindow(runwayPts, tCur, tSegEnd, vTaxiBase);
      merged = seg;
      tCur = tSegEnd;
    }
    if (retPts.length >= 2 && phyRet > 1e-9) {
      const tSegEnd = tCur + phyRet * scale;
      const seg = polylineTimelineLinearRetSpeed(retPts, tCur, tSegEnd, vRetIn, vRetOut);
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
    const { vTaxiBase, runwayPts, retPts, taxiPts, phyRw, phyRet, phyTaxi, useRwPhy, runwayLenM, vTd, vRetIn, vRetOut, carryAfterRunway } = parts;
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
        ? polylineTimelineLinearRetSpeed(runwayPts, tCur, tEnd, vTd, vRetIn)
        : polylineSpeedScaledToWindow(runwayPts, tCur, tEnd, vTaxiBase);
      merged = seg;
      tCur = tEnd;
    }
    if (retPts.length >= 2 && phyRet > 1e-9) {
      const tEnd = tCur + phyRet * scale;
      const seg = polylineTimelineLinearRetSpeed(retPts, tCur, tEnd, vRetIn, vRetOut);
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
    const toLineupOrig = split.toLineup;
    const totalLen = polylineTotalLength(toLineupOrig);
    const lineupPt = toLineupOrig[toLineupOrig.length - 1];
    const runwayId = f.depRunwayId || (f.token && (f.token.depRunwayId != null ? f.token.depRunwayId : f.token.runwayId)) || f.arrRunwayId;
    const rwTw = (state.taxiways || []).find(function(t) { return t && t.id === runwayId && t.pathType === 'runway'; });
    const exp = rwTw ? expandRtxCandidateIdsTouchingLineup(rwTw, lineupPt) : { allIds: new Set() };
    const holdPick = findLastRunwayHoldingOnDeparturePath(toLineupOrig, exp.allIds);
    const alongCut = Math.max(1e-6, totalLen);
    const backClamped = 0;
    const splitCut = polylineSplitAtDistance(toLineupOrig, alongCut);
    let pathToQueue = (splitCut.first && splitCut.first.length >= 2) ? splitCut.first : toLineupOrig;
    if (pathToQueue.length < 2) pathToQueue = toLineupOrig;
    const distHold = (holdPick && holdPick.distAlong > 1e-3 && holdPick.distAlong < alongCut - 1e-3) ? holdPick.distAlong : -1;
    let p1 = null, p2 = null;
    if (distHold > 0) {
      const splH = polylineSplitAtDistance(toLineupOrig, distHold);
      p1 = splH.first && splH.first.length >= 2 ? splH.first : null;
      const rest = splH.second;
      if (rest && rest.length >= 2 && alongCut > distHold + 1e-6) {
        const splQ = polylineSplitAtDistance(rest, alongCut - distHold);
        p2 = splQ.first && splQ.first.length >= 2 ? splQ.first : null;
      }
    }
    const validHold = !!(p1 && p2 && p1.length >= 2 && p2.length >= 2);
    let tau1 = 0, tau2 = 0;
    if (validHold) {
      tau1 = polylineDurationSecTaxi(p1);
      tau2 = polylineDurationSecTaxi(p2);
    }
    const tauSum = tau1 + tau2;
    const makeVelTaxi = makeTaxiSegmentVelocityCallback();
    const accelRoll = depTakeoffAccelMs2ForFlight(f);
    const lastQ = pathToQueue[pathToQueue.length - 1];
    const lx0 = lastQ[0], ly0 = lastQ[1];
    let runwayTailAdj = split.runwayTail;
    if (!(t3 > t0 + eps)) {
      const tl = [{ t: t0, x: lx0, y: ly0 }, { t: t3, x: lx0, y: ly0 }];
      const depRotFull = (typeof computeDepRotSecondsForFlight === 'function') ? computeDepRotSecondsForFlight(f) : Math.max(0, t3 - t0);
      return {
        timeline: tl,
        meta: {
          eobtSec: t0, etotSec: t3,
          depTaxiLineupSec: 0, depTaxiDelaySec: 0, depTaxiLineupSecReq: depTaxiLineupSecReq, depTaxiDelaySecReq: depTaxiDelaySecReq,
          lineupArrivalSec: t0, depRollStartSec: t0, depRotSec: depRotFull, depLineupHoldSec: 0, depTaxiDelayAtHolding: false,
          lineupBackM: backClamped,
        },
      };
    }
    const maxSpan = t3 - t0 - eps;
    let taxiSecUsed = Math.min(depTaxiLineupSecReq, maxSpan);
    let tAfterTaxi = t0 + taxiSecUsed;
    let afterTaxi = Math.max(0, t3 - tAfterTaxi - eps);
    let delaySecUsed = Math.min(depTaxiDelaySecReq, afterTaxi);
    let tAfterDelay = tAfterTaxi + delaySecUsed;
    let afterDelay = Math.max(0, t3 - tAfterDelay - eps);
    let lineupHoldSec = Math.min(DEP_LINEUP_HOLD_SEC, afterDelay);
    let merged;
    let t_cur = t0;
    if (validHold) {
      const r1 = tauSum > 1e-6 ? (tau1 / tauSum) : 1;
      const t1dur = taxiSecUsed * r1;
      const t2dur = taxiSecUsed * (1 - r1);
      const taxiTl1 = polylineTimelineBySegmentSpeeds(p1, t_cur, t_cur + t1dur, makeVelTaxi);
      t_cur += t1dur;
      const lastP1 = p1[p1.length - 1];
      const delayTl = (delaySecUsed > eps) ? [{ t: t_cur, x: lastP1[0], y: lastP1[1] }, { t: t_cur + delaySecUsed, x: lastP1[0], y: lastP1[1] }] : [];
      t_cur += delaySecUsed;
      const taxiTl2 = polylineTimelineBySegmentSpeeds(p2, t_cur, t_cur + t2dur, makeVelTaxi);
      t_cur += t2dur;
      merged = mergeTimelineSegments(taxiTl1, delayTl);
      merged = mergeTimelineSegments(merged, taxiTl2);
    } else {
      const taxiTl = polylineTimelineBySegmentSpeeds(pathToQueue, t0, t0 + taxiSecUsed, makeVelTaxi);
      t_cur = t0 + taxiSecUsed;
      const delayTl = (delaySecUsed > eps) ? [{ t: t_cur, x: lx0, y: ly0 }, { t: t_cur + delaySecUsed, x: lx0, y: ly0 }] : [];
      t_cur += delaySecUsed;
      merged = mergeTimelineSegments(taxiTl, delayTl);
    }
    const lastT = merged[merged.length - 1];
    const lx = lastT.x, ly = lastT.y;
    const tAtQueue = lastT.t;
    if (runwayTailAdj && runwayTailAdj.length >= 2 && dist2(runwayTailAdj[0], [lx, ly]) > 1e-4) {
      runwayTailAdj = [[lx, ly]].concat(runwayTailAdj.slice());
    }
    const tRollStart = tAtQueue + lineupHoldSec;
    const lineupHoldTl = (lineupHoldSec > eps) ? [{ t: tAtQueue, x: lx, y: ly }, { t: tRollStart, x: lx, y: ly }] : [];
    let rollTl;
    if (runwayTailAdj && runwayTailAdj.length >= 2 && t3 > tRollStart + eps) {
      rollTl = polylineTimelineConstantAccelFromRest(runwayTailAdj, tRollStart, t3, accelRoll);
    } else {
      rollTl = [{ t: tRollStart, x: lx, y: ly }, { t: t3, x: lx, y: ly }];
    }
    merged = mergeTimelineSegments(merged, lineupHoldTl);
    merged = mergeTimelineSegments(merged, rollTl);
    const rollWindow = Math.max(0, t3 - tRollStart);
    const depRotFullSec = (typeof computeDepRotSecondsForFlight === 'function') ? computeDepRotSecondsForFlight(f) : (lineupHoldSec + rollWindow);
    return {
      timeline: merged,
      meta: {
        eobtSec: t0, etotSec: t3,
        depTaxiLineupSec: taxiSecUsed, depTaxiDelaySec: delaySecUsed,
        depTaxiLineupSecReq: depTaxiLineupSecReq, depTaxiDelaySecReq: depTaxiDelaySecReq,
        lineupArrivalSec: tAfterTaxi, depRollStartSec: tRollStart,
        depRotSec: depRotFullSec, depLineupHoldSec: lineupHoldSec,
        depTaxiDelayAtHolding: validHold,
        lineupBackM: backClamped,
      },
    };
  }
  function buildFullAirsideTimelineForFlight(f) {
    if (!f) return;
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
    if (flightBlockedLikeNoWay(f)) {
      f.timeline = null;
      f.timeline_meta = { error: (f.arrDep !== 'Dep' && f.arrRetFailed && !f.noWayArr && !f.noWayDep) ? 'arr_ret_failed' : 'no_path' };
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
    const builtAppr = buildArrivalApproachPolylinePts(runwayId, rwDir, anchorDist, offset, tdPt);
    const pack = builtAppr.pack;
    const apprPts = builtAppr.apprPts;
    if (!apprPts || apprPts.length < 2) {
      f.timeline = null;
      f.timeline_meta = { error: 'no_appr' };
      return;
    }
    const rawApprDur = polylineRawDurationSegmentVelocities(apprPts, function() { return vTd; });
    const t0 = eldtS - rawApprDur;
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
        { t: runwayEndT, x: lastRw.x, y: lastRw.y },
        { t: tTaxiStart, x: lastRw.x, y: lastRw.y },
      ]);
    }
    const apronTl = buildApronTaxiTimelineAfterRet(f, runwayId, taxiInPts, tTaxiStart, eibtS);
    taxiInTl = mergeTimelineSegments(taxiInTl, apronTl);
    const standPt = taxiInPts.length ? taxiInPts[taxiInPts.length - 1] : arrPts[arrPts.length - 1];
    const sx = standPt[0], sy = standPt[1];
    const dwellTl = [{ t: eibtS, x: sx, y: sy }, { t: eobtS, x: sx, y: sy }];
    const builtDep = buildDepartureSurfaceTimelineSegments(f, eobtS, etotS);
    if (!builtDep || !builtDep.timeline || builtDep.timeline.length < 2) {
      f.timeline = null;
      f.timeline_meta = { error: 'no_path', leg: 'dep_tail' };
      return;
    }
    const depTl = builtDep.timeline;
    let timeline = mergeTimelineSegments(airTl, taxiInTl);
    timeline = mergeTimelineSegments(timeline, dwellTl);
    timeline = mergeTimelineSegments(timeline, depTl);
    f.timeline = timeline;
    f.timeline_meta = Object.assign({
      tApproachStart: t0,
      eldtSec: eldtS,
      eibtSec: eibtS,
      eobtSec: eobtS,
      etotSec: etotS,
      approachOffset: offset,
      approachStraightFinalM: APPROACH_STRAIGHT_FINAL_M,
      approachPathLenM: (pack && typeof pack.pathLen === 'number') ? pack.pathLen : null,
      touchdownSpeedMs: vTd,
    }, builtDep.meta || {});
  }
  let _lazyTimelineLastEvictSimSec = NaN;
  function clearAllFlightTimelines(opts) {
    const keepDes = opts && opts.keepDesResultTimelines === true;
    const flights = state.flights || [];
    for (let i = 0; i < flights.length; i++) {
      const f = flights[i];
      if (!f) continue;
      if (keepDes && f.timeline_meta && f.timeline_meta.playbackSource === 'des_result') continue;
      f.timeline = null;
      delete f.timeline_meta;
      if (!keepDes) {
        delete f.proSimEdgeList;
        delete f.edge_list;
      }
    }
    _lazyTimelineLastEvictSimSec = NaN;
  }
  function prepareLazyTimelinesForCurrentSim(tSec) {
    const flights = state.flights || [];
    const pad = simAirsideLazyPadSec();
    const tEvictKey = Number(tSec);
    if (!isFinite(tEvictKey) || tEvictKey !== _lazyTimelineLastEvictSimSec) {
      if (isFinite(tEvictKey)) _lazyTimelineLastEvictSimSec = tEvictKey;
      for (let i = 0; i < flights.length; i++) {
        const f = flights[i];
        if (!f) continue;
        if (flightBlockedLikeNoWay(f)) continue;
        if (!f.timeline || !f.timeline.length) continue;
        const meta = f.timeline_meta;
        if (meta && meta.playbackSource === 'des_result') continue;
        const w = getFlightAirsideWindowSec(f);
        if (!w) { f.timeline = null; continue; }
        if (tSec > w.t1 + 1e-3 || tSec < w.t0 - pad - 1e-3) f.timeline = null;
      }
    }
    const pending = [];
    for (let i = 0; i < flights.length; i++) {
      const f = flights[i];
      if (!f) continue;
      if (flightBlockedLikeNoWay(f)) continue;
      if (!isFlightAirsideLazyTimelineBuildEligible(f, tSec)) continue;
      if (f.timeline && f.timeline.length) continue;
      pending.push(f);
    }
    if (!pending.length) return;
    const tN = Number(tSec);
    const so = state.selectedObject;
    pending.sort(function(a, b) {
      const selA = so && so.type === 'flight' && so.id === a.id;
      const selB = so && so.type === 'flight' && so.id === b.id;
      if (selA !== selB) return selA ? -1 : 1;
      const actA = isFlightAirsideActiveAtSimSec(a, tN) ? 0 : 1;
      const actB = isFlightAirsideActiveAtSimSec(b, tN) ? 0 : 1;
      if (actA !== actB) return actA - actB;
      const wa = getFlightAirsideWindowSec(a);
      const wb = getFlightAirsideWindowSec(b);
      const da = wa ? Math.abs(tN - (wa.t0 + wa.t1) * 0.5) : Infinity;
      const db = wb ? Math.abs(tN - (wb.t0 + wb.t1) * 0.5) : Infinity;
      if (da !== db) return da - db;
      return (wa && wb) ? (wa.t0 - wb.t0) : 0;
    });
    const cap = MAX_LAZY_TIMELINE_BUILDS_PER_FRAME;
    for (let k = 0; k < pending.length && k < cap; k++) {
      buildFullAirsideTimelineForFlight(pending[k]);
    }
  }
  function rebuildAllFlightAirsideTimelines() {
    clearAllFlightTimelines();
  }

  
  function getRunwayLineupDistMByDirection(tw, dir) {
    if (!tw || tw.pathType !== 'runway') return 0;
    const isCcw = normalizeRwDirectionValue(dir) === 'counter_clockwise';
    const primary = isCcw ? tw.lineupDistM_CCW : tw.lineupDistM_CW;
    if (typeof primary === 'number' && isFinite(primary) && primary >= 0) return primary;
    const legacy = tw.lineupDistM;
    if (typeof legacy === 'number' && isFinite(legacy) && legacy >= 0) return legacy;
    return 0;
  }
  function getEffectiveRunwayLineupDistM(tw) {
    if (!tw || tw.pathType !== 'runway') return 0;
    return getRunwayLineupDistMByDirection(tw, getTaxiwayDirection(tw));
  }
  function getEffectiveRunwayLineupDistFromStartM(tw, runwayLenM) {
    if (!tw || tw.pathType !== 'runway') return 0;
    const len = (typeof runwayLenM === 'number' && isFinite(runwayLenM) && runwayLenM >= 0) ? runwayLenM : 0;
    const d = getRunwayLineupDistMByDirection(tw, getTaxiwayDirection(tw));
    return Math.max(0, Math.min(len, d));
  }

  function getEffectiveRunwayStartDisplacedThresholdM(tw) {
    if (!tw || tw.pathType !== 'runway') return RUNWAY_START_DISPLACED_THRESHOLD_DEFAULT_M;
    const v = tw.startDisplacedThresholdM;
    return (typeof v === 'number' && isFinite(v) && v >= 0) ? v : RUNWAY_START_DISPLACED_THRESHOLD_DEFAULT_M;
  }

  function getEffectiveRunwayStartBlastPadM(tw) {
    if (!tw || tw.pathType !== 'runway') return RUNWAY_START_BLAST_PAD_DEFAULT_M;
    const v = tw.startBlastPadM;
    return (typeof v === 'number' && isFinite(v) && v >= 0) ? v : RUNWAY_START_BLAST_PAD_DEFAULT_M;
  }

  function getEffectiveRunwayEndDisplacedThresholdM(tw) {
    if (!tw || tw.pathType !== 'runway') return RUNWAY_END_DISPLACED_THRESHOLD_DEFAULT_M;
    const v = tw.endDisplacedThresholdM;
    return (typeof v === 'number' && isFinite(v) && v >= 0) ? v : RUNWAY_END_DISPLACED_THRESHOLD_DEFAULT_M;
  }

  function getEffectiveRunwayEndBlastPadM(tw) {
    if (!tw || tw.pathType !== 'runway') return RUNWAY_END_BLAST_PAD_DEFAULT_M;
    const v = tw.endBlastPadM;
    return (typeof v === 'number' && isFinite(v) && v >= 0) ? v : RUNWAY_END_BLAST_PAD_DEFAULT_M;
  }

  function runwayPolylineLengthPx(pts) {
    if (!pts || pts.length < 2) return 0;
    let s = 0;
    for (let i = 0; i < pts.length - 1; i++) s += pathDist(pts[i], pts[i + 1]);
    return s;
  }

  
  function runwayApproachThresholdDistAlongM(runwayId, tdDistAlong) {
    const path = getRunwayPath(runwayId);
    if (!path || !path.pts || path.pts.length < 2) return 0;
    const totalLen = runwayPolylineLengthPx(path.pts);
    const tw = (state.taxiways || []).find(function(t) { return t && t.id === runwayId && t.pathType === 'runway'; });
    if (!tw) return 0;
    const startInset = getEffectiveRunwayStartDisplacedThresholdM(tw) + getEffectiveRunwayStartBlastPadM(tw);
    const endInset = getEffectiveRunwayEndDisplacedThresholdM(tw) + getEffectiveRunwayEndBlastPadM(tw);
    const isCcw = normalizeRwDirectionValue(getTaxiwayDirection(tw)) === 'counter_clockwise';
    const lowInset = isCcw ? endInset : startInset;
    const highInset = isCcw ? startInset : endInset;
    const dLow = Math.min(Math.max(0, lowInset), totalLen);
    const dHigh = Math.max(0, Math.min(totalLen, totalLen - highInset));
    if (!(totalLen > 1e-6)) return dLow;
    if (tdDistAlong <= totalLen * 0.5) return dLow;
    return dHigh;
  }

  function getPolylinePointAndFrameAtDistance(pts, distPx) {
    if (!pts || pts.length < 2) return null;
    const total = runwayPolylineLengthPx(pts);
    const d = Math.max(0, Math.min(typeof distPx === 'number' ? distPx : 0, total));
    let acc = 0;
    for (let i = 0; i < pts.length - 1; i++) {
      const p1 = pts[i], p2 = pts[i + 1];
      const segLen = pathDist(p1, p2);
      if (!(segLen > 1e-6)) continue;
      if (acc + segLen >= d - 1e-6) {
        const t = Math.max(0, Math.min(1, (d - acc) / segLen));
        const ux = (p2[0] - p1[0]) / segLen;
        const uy = (p2[1] - p1[1]) / segLen;
        return {
          point: [p1[0] + (p2[0] - p1[0]) * t, p1[1] + (p2[1] - p1[1]) * t],
          tangent: [ux, uy],
          normal: [-uy, ux]
        };
      }
      acc += segLen;
    }
    const last = pts[pts.length - 1], prev = pts[pts.length - 2];
    const segLen = pathDist(prev, last);
    if (!(segLen > 1e-6)) return null;
    const ux = (last[0] - prev[0]) / segLen;
    const uy = (last[1] - prev[1]) / segLen;
    return { point: [last[0], last[1]], tangent: [ux, uy], normal: [-uy, ux] };
  }

  function drawRunwayDecorations(tw, pts, widthPx, opts) {
    if (!tw || tw.pathType !== 'runway') return;
    if (!pts || pts.length < 2) return;
    const baseOnly = !!(opts && opts.baseOnly);
    const markingsOnly = !!(opts && opts.markingsOnly);
    const totalLen = runwayPolylineLengthPx(pts);
    const runwayWidth = Math.max(24, Number(widthPx) || RUNWAY_PATH_DEFAULT_WIDTH);
    if (totalLen < Math.max(220, runwayWidth * 3)) return;
    const startDisp = getEffectiveRunwayStartDisplacedThresholdM(tw);
    const startBlast = getEffectiveRunwayStartBlastPadM(tw);
    const endDisp = getEffectiveRunwayEndDisplacedThresholdM(tw);
    const endBlast = getEffectiveRunwayEndBlastPadM(tw);
    const lowFrame = getPolylinePointAndFrameAtDistance(pts, 0);
    const highFrame = getPolylinePointAndFrameAtDistance(pts, totalLen);
    if (!lowFrame || !highFrame) return;
    const isCcw = normalizeRwDirectionValue(getTaxiwayDirection(tw)) === 'counter_clockwise';
    const startFrame = isCcw ? highFrame : lowFrame;
    const endFrame = isCcw ? lowFrame : highFrame;
    const startSegSign = isCcw ? 1 : -1;
    const endSegSign = isCcw ? -1 : 1;
    const startArrowPos = isCcw ? 1 : -1;
    const startArrowDir = isCcw ? -1 : 1;
    const endArrowPos = isCcw ? -1 : 1;
    const endArrowDir = isCcw ? 1 : -1;

    function drawRectWithFrame(frame, alongOffsetPx, lateralOffsetPx, alongLenPx, acrossLenPx, fillStyle, strokeStyle, lineWidth) {
      if (!frame) return;
      const cx = frame.point[0] + frame.tangent[0] * alongOffsetPx + frame.normal[0] * lateralOffsetPx;
      const cy = frame.point[1] + frame.tangent[1] * alongOffsetPx + frame.normal[1] * lateralOffsetPx;
      const hx = frame.tangent[0] * alongLenPx * 0.5;
      const hy = frame.tangent[1] * alongLenPx * 0.5;
      const wx = frame.normal[0] * acrossLenPx * 0.5;
      const wy = frame.normal[1] * acrossLenPx * 0.5;
      ctx.beginPath();
      ctx.moveTo(cx - hx - wx, cy - hy - wy);
      ctx.lineTo(cx + hx - wx, cy + hy - wy);
      ctx.lineTo(cx + hx + wx, cy + hy + wy);
      ctx.lineTo(cx - hx + wx, cy - hy + wy);
      ctx.closePath();
      if (fillStyle) {
        ctx.fillStyle = fillStyle;
        ctx.fill();
      }
      if (strokeStyle && lineWidth > 0) {
        ctx.lineWidth = lineWidth;
        ctx.strokeStyle = strokeStyle;
        ctx.stroke();
      }
    }

    function drawRectAtDistance(distPx, lateralOffsetPx, alongLenPx, acrossLenPx, fillStyle) {
      const frame = getPolylinePointAndFrameAtDistance(pts, distPx);
      if (!frame) return;
      drawRectWithFrame(frame, 0, lateralOffsetPx, alongLenPx, acrossLenPx, fillStyle, null, 0);
    }

    function drawRectAtBothEnds(distPx, lateralOffsetPx, alongLenPx, acrossLenPx, fillStyle) {
      if (!(distPx > 0) || distPx >= totalLen - 1) return;
      drawRectAtDistance(distPx, lateralOffsetPx, alongLenPx, acrossLenPx, fillStyle);
      drawRectAtDistance(totalLen - distPx, lateralOffsetPx, alongLenPx, acrossLenPx, fillStyle);
    }

    function drawSymmetricPairAtBothEnds(distPx, lateralOffsetPx, alongLenPx, acrossLenPx, fillStyle) {
      drawRectAtBothEnds(distPx, lateralOffsetPx, alongLenPx, acrossLenPx, fillStyle);
      if (Math.abs(lateralOffsetPx) > 1e-6) {
        drawRectAtBothEnds(distPx, -lateralOffsetPx, alongLenPx, acrossLenPx, fillStyle);
      }
    }

    ctx.save();
    const rwDecoOpaque = !!state.layers.pathFill;
    const rwPaveAsphalt = c2dRoadWidthBandRunwayAsphaltColor();
    const monoFill = layerMonoFillOn() && rwDecoOpaque;
    const monoLine = layerMonoLinesOn() && !!state.layers.pathLines;
    const monoFillCss = c2dLayerMonoFillDarkAsphaltCss();
    const monoLineCss = c2dLayerMonoLineStrokeCss();
    function rwO(c) { return rwDecoOpaque ? c2dCssColorToOpaque(c) : c; }
    const thresholdColor = monoLine ? monoLineCss : rwO(c2dRunwayThresholdColor());
    const displacedArrowFill = monoLine ? monoLineCss : (rwDecoOpaque ? c2dCssColorToOpaque(c2dRunwayMarkingColor()) : thresholdColor);
    const touchdownColor = monoLine ? monoLineCss : rwO(c2dRunwayTouchdownColor());
    const aimingPointColor = monoLine ? monoLineCss : rwO(c2dRunwayAimingPointColor());
    const extensionFill = monoFill ? monoFillCss : (rwDecoOpaque ? rwPaveAsphalt : rwO(c2dRunwayExtensionFill()));
    const extensionOutline = monoLine ? monoLineCss : c2dRunwayOutline();
    const blastChevronColor = monoLine ? monoLineCss : (rwDecoOpaque ? c2dCssColorToOpaque(c2dRunwayBlastChevronColor()) : rwO(c2dRunwayBlastChevronColor()));

    function drawExtensionSegment(frame, directionSign, innerOffsetPx, segLenPx) {
      if (!(segLenPx > 0)) return;
      drawRectWithFrame(
        frame,
        directionSign * (innerOffsetPx + segLenPx * 0.5),
        0,
        segLenPx,
        runwayWidth,
        extensionFill,
        extensionOutline,
        1.2
      );
    }

    function drawDisplacedThresholdArrows(frame, positionSign, arrowDirectionSign, innerOffsetPx, segLenPx) {
      if (!(segLenPx > 0)) return;
      const count = Math.max(2, Math.min(8, Math.round(segLenPx / 30)));
      const arrowSpan = Math.min(Math.max(segLenPx * 0.22, runwayWidth * 0.42), segLenPx * 0.42);
      const usableLen = Math.max(0, segLenPx - arrowSpan);
      const shaftHalf = Math.max(3, runwayWidth * 0.055);
      const headLen = Math.min(Math.max(16, arrowSpan * 0.32), arrowSpan * 0.48);
      ctx.fillStyle = displacedArrowFill;
      for (let i = 0; i < count; i++) {
        const along = innerOffsetPx + (arrowSpan * 0.5) + (usableLen * (i + 0.5) / count);
        const framePoint = [frame.point[0] + frame.tangent[0] * positionSign * along, frame.point[1] + frame.tangent[1] * positionSign * along];
        const tipX = framePoint[0] + frame.tangent[0] * arrowDirectionSign * (arrowSpan * 0.5);
        const tipY = framePoint[1] + frame.tangent[1] * arrowDirectionSign * (arrowSpan * 0.5);
        const tailX = framePoint[0] - frame.tangent[0] * arrowDirectionSign * (arrowSpan * 0.5);
        const tailY = framePoint[1] - frame.tangent[1] * arrowDirectionSign * (arrowSpan * 0.5);
        const neckX = tipX - frame.tangent[0] * arrowDirectionSign * headLen;
        const neckY = tipY - frame.tangent[1] * arrowDirectionSign * headLen;
        const halfWidth = Math.max(7, runwayWidth * 0.13);
        ctx.beginPath();
        ctx.moveTo(tailX - frame.normal[0] * shaftHalf, tailY - frame.normal[1] * shaftHalf);
        ctx.lineTo(neckX - frame.normal[0] * shaftHalf, neckY - frame.normal[1] * shaftHalf);
        ctx.lineTo(neckX - frame.normal[0] * halfWidth, neckY - frame.normal[1] * halfWidth);
        ctx.lineTo(tipX, tipY);
        ctx.lineTo(neckX + frame.normal[0] * halfWidth, neckY + frame.normal[1] * halfWidth);
        ctx.lineTo(neckX + frame.normal[0] * shaftHalf, neckY + frame.normal[1] * shaftHalf);
        ctx.lineTo(tailX + frame.normal[0] * shaftHalf, tailY + frame.normal[1] * shaftHalf);
        ctx.closePath();
        ctx.fill();
      }
    }

    function drawBlastPadChevrons(frame, positionSign, innerOffsetPx, segLenPx) {
      if (!(segLenPx > 0)) return;
      const count = Math.max(2, Math.min(7, Math.round(segLenPx / 35)));
      const sideReach = Math.max(12, runwayWidth * 0.46);
      const chevronDepth = Math.max(14, sideReach / Math.tan(Math.PI / 3));
      const usableLen = Math.max(0, segLenPx - chevronDepth);
      ctx.save();
      ctx.lineWidth = Math.max(3, runwayWidth * 0.075);
      ctx.lineCap = 'square';
      ctx.lineJoin = 'miter';
      ctx.strokeStyle = blastChevronColor;
      for (let i = 0; i < count; i++) {
        const along = innerOffsetPx + (chevronDepth * 0.5) + (usableLen * (i + 0.5) / count);
        const apexX = frame.point[0] + frame.tangent[0] * positionSign * along;
        const apexY = frame.point[1] + frame.tangent[1] * positionSign * along;
        const outerAlong = along + chevronDepth;
        const leftX = frame.point[0] + frame.tangent[0] * positionSign * outerAlong + frame.normal[0] * sideReach;
        const leftY = frame.point[1] + frame.tangent[1] * positionSign * outerAlong + frame.normal[1] * sideReach;
        const rightX = frame.point[0] + frame.tangent[0] * positionSign * outerAlong - frame.normal[0] * sideReach;
        const rightY = frame.point[1] + frame.tangent[1] * positionSign * outerAlong - frame.normal[1] * sideReach;
        ctx.beginPath();
        ctx.moveTo(leftX, leftY);
        ctx.lineTo(apexX, apexY);
        ctx.lineTo(rightX, rightY);
        ctx.stroke();
      }
      ctx.restore();
    }

    if (!markingsOnly) {
      drawExtensionSegment(startFrame, startSegSign, 0, startDisp);
      drawExtensionSegment(startFrame, startSegSign, startDisp, startBlast);
      drawExtensionSegment(endFrame, endSegSign, 0, endDisp);
      drawExtensionSegment(endFrame, endSegSign, endDisp, endBlast);
    }
    if (!baseOnly) {
      drawDisplacedThresholdArrows(startFrame, startArrowPos, startArrowDir, 0, startDisp);
      drawDisplacedThresholdArrows(endFrame, endArrowPos, endArrowDir, 0, endDisp);
      drawBlastPadChevrons(startFrame, startSegSign, startDisp, startBlast);
      drawBlastPadChevrons(endFrame, endSegSign, endDisp, endBlast);
    }

    const thresholdInset = Math.min(Math.max(runwayWidth * 0.58, 26), totalLen * 0.12);
    const thresholdStripeLen = Math.min(Math.max(runwayWidth * 0.54, 20), 34);
    const thresholdStripeWidth = Math.max(3, runwayWidth * 0.085);
    if (!baseOnly) {
      [-runwayWidth * 0.30, -runwayWidth * 0.18, -runwayWidth * 0.06, runwayWidth * 0.06, runwayWidth * 0.18, runwayWidth * 0.30].forEach(function(offset) {
        drawRectAtBothEnds(thresholdInset, offset, thresholdStripeLen, thresholdStripeWidth, thresholdColor);
      });
    }

    const aimingDist = Math.min(Math.max(300, runwayWidth * 3.5), totalLen * 0.28);
    if (!baseOnly) {
      if (aimingDist < (totalLen * 0.5) - (runwayWidth * 0.6)) {
        drawSymmetricPairAtBothEnds(
          aimingDist,
          runwayWidth * 0.20,
          Math.min(Math.max(runwayWidth * 1.2, 54), 92),
          Math.max(6, runwayWidth * 0.12),
          aimingPointColor
        );
      }
    }

    if (!baseOnly) {
      [150, 450].forEach(function(distPx) {
        if (distPx >= (totalLen * 0.5) - (runwayWidth * 0.8)) return;
        [runwayWidth * 0.14, runwayWidth * 0.28].forEach(function(offsetPx) {
          drawSymmetricPairAtBothEnds(
            distPx,
            offsetPx,
            Math.min(Math.max(runwayWidth * 0.52, 22), 42),
            Math.max(4, runwayWidth * 0.08),
            touchdownColor
          );
        });
      });
    }
    ctx.restore();
  }

  /** Paved-segment runway centerline; drawn after all taxiway width strokes so RTX width cannot cover it. */
  function drawRunwayPavedCenterlineDashed(tw, pts, widthPx) {
    if (!tw || tw.pathType !== 'runway') return;
    if (!pts || pts.length < 2) return;
    const totalLen = runwayPolylineLengthPx(pts);
    const runwayWidth = Math.max(24, Number(widthPx) || RUNWAY_PATH_DEFAULT_WIDTH);
    if (totalLen < Math.max(220, runwayWidth * 3)) return;
    const startDisp = getEffectiveRunwayStartDisplacedThresholdM(tw);
    const startBlast = getEffectiveRunwayStartBlastPadM(tw);
    const endDisp = getEffectiveRunwayEndDisplacedThresholdM(tw);
    const endBlast = getEffectiveRunwayEndBlastPadM(tw);
    const lowFrame = getPolylinePointAndFrameAtDistance(pts, 0);
    const highFrame = getPolylinePointAndFrameAtDistance(pts, totalLen);
    if (!lowFrame || !highFrame) return;
    const isCcw = normalizeRwDirectionValue(getTaxiwayDirection(tw)) === 'counter_clockwise';
    const paveStart = isCcw ? (endDisp + endBlast) : (startDisp + startBlast);
    const paveEnd = isCcw ? (totalLen - startDisp - startBlast) : (totalLen - endDisp - endBlast);
    if (!(paveEnd > paveStart + 1)) return;
    const clPts = polylineSliceBetweenDistances(pts, paveStart, paveEnd);
    if (!clPts || clPts.length < 2) return;
    ctx.save();
    ctx.strokeStyle = layerMonoLinesOn() && state.layers.pathLines ? c2dLayerMonoLineStrokeCss() : c2dRunwayCenterlineColor();
    ctx.lineWidth = Math.max(1, runwayWidth * 0.02);
    const dashPx = Math.max(10, runwayWidth * 0.2);
    const gapPx = Math.max(8, runwayWidth * 0.16);
    ctx.setLineDash([dashPx, gapPx]);
    ctx.lineDashOffset = 0;
    ctx.lineCap = 'butt';
    ctx.lineJoin = 'miter';
    ctx.beginPath();
    ctx.moveTo(clPts[0][0], clPts[0][1]);
    for (let ci = 1; ci < clPts.length; ci++) ctx.lineTo(clPts[ci][0], clPts[ci][1]);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();
  }

  function polylineTailFromDistancePx(pts, distPx) {
    if (!pts || pts.length < 2) return [];
    const total = runwayPolylineLengthPx(pts);
    const d = Math.max(0, Math.min(distPx, total));
    if (d <= 1e-9) return pts.map(p => [p[0], p[1]]);
    let acc = 0;
    for (let i = 0; i < pts.length - 1; i++) {
      const p1 = pts[i], p2 = pts[i + 1];
      const segLen = pathDist(p1, p2);
      if (segLen < 1e-9) continue;
      if (acc + segLen >= d - 1e-6) {
        const t = Math.max(0, Math.min(1, (d - acc) / segLen));
        const lp = [p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1])];
        const out = [lp];
        for (let j = i + 1; j < pts.length; j++) out.push([pts[j][0], pts[j][1]]);
        return out;
      }
      acc += segLen;
    }
    return [[pts[pts.length - 1][0], pts[pts.length - 1][1]]];
  }

  function polylineSliceBetweenDistances(pts, d0, d1) {
    if (!pts || pts.length < 2) return [];
    const total = runwayPolylineLengthPx(pts);
    let a = Math.max(0, Math.min(typeof d0 === 'number' ? d0 : 0, total));
    let b = Math.max(a, Math.min(typeof d1 === 'number' ? d1 : total, total));
    if (b - a < 1e-6) return [];
    function pointAtDist(d) {
      let acc = 0;
      for (let i = 0; i < pts.length - 1; i++) {
        const p1 = pts[i], p2 = pts[i + 1];
        const segLen = pathDist(p1, p2);
        if (segLen < 1e-9) continue;
        if (acc + segLen >= d - 1e-6) {
          const t = Math.max(0, Math.min(1, (d - acc) / segLen));
          return { pt: [p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1])], segIndex: i };
        }
        acc += segLen;
      }
      const last = pts[pts.length - 1];
      return { pt: [last[0], last[1]], segIndex: Math.max(0, pts.length - 2) };
    }
    const start = pointAtDist(a);
    const end = pointAtDist(b);
    const out = [[start.pt[0], start.pt[1]]];
    if (start.segIndex === end.segIndex) {
      if (dist2(start.pt, end.pt) > 1e-9) out.push([end.pt[0], end.pt[1]]);
      return out;
    }
    for (let si = start.segIndex + 1; si <= end.segIndex; si++) {
      if (si < pts.length) out.push([pts[si][0], pts[si][1]]);
    }
    if (dist2(out[out.length - 1], end.pt) > 1e-9) out.push([end.pt[0], end.pt[1]]);
    return dedupePathPoints(out);
  }

  function syncStartEndFromVertices(obj) {
    if (!obj || !obj.vertices || obj.vertices.length < 2) return;
    const first = obj.vertices[0], last = obj.vertices[obj.vertices.length - 1];
    obj.start_point = { col: first.col, row: first.row };
    obj.end_point = { col: last.col, row: last.row };
  }
  /** CW↔CCW 전환 시 vertices 반전 → vertices[0]이 현재 direction 기준 시작점, lineup은 해당 모드 거리로 표시. */
  function runwayReverseVerticesIfDirectionChanged(tw, nextDirRaw) {
    if (!tw || tw.pathType !== 'runway' || !tw.vertices || tw.vertices.length < 2) return;
    const prevNorm = (tw.direction === 'counter_clockwise') ? 'counter_clockwise' : 'clockwise';
    const nextNorm = (String(nextDirRaw || '').trim() === 'counter_clockwise') ? 'counter_clockwise' : 'clockwise';
    if (prevNorm === nextNorm) return;
    tw.vertices.reverse();
    syncStartEndFromVertices(tw);
  }
  function getTaxiwayOrderedPoints(tw) {
    if (!tw.vertices || tw.vertices.length < 2) return null;
    const pts = tw.vertices.map(v => cellToPixel(v.col, v.row));
    return pts;
  }
  function getOrderedPoints(obj) {
    if (!obj || !obj.vertices || obj.vertices.length < 2) return null;
    const isRunway = obj.pathType === 'runway';
    if (isRunway) { const r = getRunwayPath(obj.id); return r && r.pts ? r.pts : null; }
    return getTaxiwayOrderedPoints(obj);
  }

  function projectOnSegment(a, b, q) {
    const ax = a[0], ay = a[1], bx = b[0], by = b[1], qx = q[0], qy = q[1];
    const dx = bx - ax, dy = by - ay, den = dx*dx + dy*dy;
    if (den < 1e-12) return { t: 0, p: a };
    let t = ((qx-ax)*dx + (qy-ay)*dy) / den;
    t = Math.max(0, Math.min(1, t));
    return { t, p: [ax+t*dx, ay+t*dy] };
  }
  function segmentSegmentIntersection(a, b, c, d) {
    const ax = a[0], ay = a[1], bx = b[0], by = b[1];
    const cx = c[0], cy = c[1], dx = d[0], dy = d[1];
    const rx = bx - ax, ry = by - ay, sx = dx - cx, sy = dy - cy;
    const cross = rx * sy - ry * sx;
    if (Math.abs(cross) < 1e-12) return null;
    const t = ((cx - ax) * sy - (cy - ay) * sx) / cross;
    const s = ((cx - ax) * ry - (cy - ay) * rx) / cross;
    if (t < 0 || t > 1 || s < 0 || s > 1) return null;
    return { p: [ax + t * rx, ay + t * ry] };
  }
  function collinearSegmentOverlapOnAB(a, b, c, d) {
    const ax = a[0], ay = a[1], bx = b[0], by = b[1];
    const dx = bx - ax, dy = by - ay;
    const len2 = dx * dx + dy * dy;
    if (len2 < 1e-12) return null;
    const len = Math.sqrt(len2);
    function perpDistAB(p) {
      return Math.abs((p[0] - ax) * dy - (p[1] - ay) * dx) / len;
    }
    const lineTol = Math.max(0.55, len * 1e-9);
    if (perpDistAB(c) > lineTol || perpDistAB(d) > lineTol) return null;
    function tOnAB(p) {
      return ((p[0] - ax) * dx + (p[1] - ay) * dy) / len2;
    }
    const tc = tOnAB(c), td = tOnAB(d);
    const lo = Math.min(tc, td), hi = Math.max(tc, td);
    const o0 = Math.max(0, lo), o1 = Math.min(1, hi);
    if (o1 < o0 - 1e-9) return null;
    return { t0: o0, t1: o1 };
  }
  const SPLIT_TOL_D2 = 0.25;
  function pointOnSegmentStrict(a, b, q) {
    const { p } = projectOnSegment(a, b, q);
    return dist2(p, q) <= SPLIT_TOL_D2;
  }
  function polylineTouchesPolylineForGraph(pts, otherOrd) {
    if (!pts || pts.length < 2 || !otherOrd || otherOrd.length < 2) return false;
    for (let seg = 0; seg < pts.length - 1; seg++) {
      const a = pts[seg], b = pts[seg + 1];
      for (let oseg = 0; oseg < otherOrd.length - 1; oseg++) {
        const c = otherOrd[oseg], d = otherOrd[oseg + 1];
        if (segmentSegmentIntersection(a, b, c, d)) return true;
        if (collinearSegmentOverlapOnAB(a, b, c, d)) return true;
        for (let k = 0; k < 2; k++) {
          const q = k === 0 ? c : d;
          if (dist2(a, q) <= SPLIT_TOL_D2 || dist2(b, q) <= SPLIT_TOL_D2) {
            const pr = projectOnSegment(a, b, q);
            if (pr.t >= 0 && pr.t <= 1) return true;
          }
        }
      }
      for (let ri = 0; ri < otherOrd.length; ri++) {
        const q = otherOrd[ri];
        if (pointOnSegmentStrict(a, b, q)) return true;
      }
    }
    return false;
  }
  function pointNearPolylineSq(p, pts, tolD2) {
    if (!p || !pts || pts.length < 2) return false;
    const lim = (typeof tolD2 === 'number' && isFinite(tolD2) && tolD2 > 0) ? tolD2 : SPLIT_TOL_D2;
    for (let i = 0; i < pts.length - 1; i++) {
      const pr = projectOnSegment(pts[i], pts[i + 1], p);
      if (pr.t >= 0 && pr.t <= 1 && dist2(pr.p, p) <= lim) return true;
    }
    return false;
  }
  function lineupHoldingTolD2(scale) {
    const s = (typeof scale === 'number' && isFinite(scale) && scale > 0) ? scale : 1;
    const basePx = (typeof PATH_JUNCTION_MERGE_RADIUS_PX === 'number' && isFinite(PATH_JUNCTION_MERGE_RADIUS_PX) && PATH_JUNCTION_MERGE_RADIUS_PX > 0)
      ? PATH_JUNCTION_MERGE_RADIUS_PX
      : 7;
    const tolPx = basePx * s;
    return Math.max(SPLIT_TOL_D2, tolPx * tolPx);
  }
  
  function isLineupPointTouchingRunwayTaxiwayOnRunway(runwayTw, lineupPt) {
    if (!runwayTw || runwayTw.pathType !== 'runway' || !lineupPt) return false;
    const rwPts = getOrderedPoints(runwayTw);
    if (!rwPts || rwPts.length < 2) return false;
    const touchD2 = lineupHoldingTolD2(1.0);
    const list = state.taxiways || [];
    for (let ti = 0; ti < list.length; ti++) {
      const tx = list[ti];
      if (tx.pathType !== 'runway_exit') continue;
      const rtxPts = getOrderedPoints(tx);
      if (!rtxPts || rtxPts.length < 2) continue;
      if (!polylineTouchesPolylineForGraph(rtxPts, rwPts) && !polylineTouchesPolylineForGraph(rwPts, rtxPts)) continue;
      if (pointNearPolylineSq(lineupPt, rtxPts, touchD2)) return true;
    }
    return false;
  }
  function listRtxTouchingLineupOnRunway(runwayTw, lineupPt) {
    const out = [];
    if (!runwayTw || runwayTw.pathType !== 'runway' || !lineupPt) return out;
    const rwPts = getOrderedPoints(runwayTw);
    if (!rwPts || rwPts.length < 2) return out;
    const touchD2 = lineupHoldingTolD2(1.0);
    const list = state.taxiways || [];
    for (let ti = 0; ti < list.length; ti++) {
      const tx = list[ti];
      if (tx.pathType !== 'runway_exit') continue;
      const rtxPts = getOrderedPoints(tx);
      if (!rtxPts || rtxPts.length < 2) continue;
      if (!polylineTouchesPolylineForGraph(rtxPts, rwPts) && !polylineTouchesPolylineForGraph(rwPts, rtxPts)) continue;
      if (pointNearPolylineSq(lineupPt, rtxPts, touchD2)) out.push(tx);
    }
    return out;
  }
  function rtxPolylinesTouch(rtxA, rtxB) {
    const pa = getOrderedPoints(rtxA);
    const pb = getOrderedPoints(rtxB);
    if (!pa || pa.length < 2 || !pb || pb.length < 2) return false;
    return polylineTouchesPolylineForGraph(pa, pb) || polylineTouchesPolylineForGraph(pb, pa);
  }
  function rtxRunwayExitNeighborIds(rtxA) {
    const ids = new Set();
    if (!rtxA || rtxA.id == null) return ids;
    ids.add(rtxA.id);
    const list = state.taxiways || [];
    for (let ti = 0; ti < list.length; ti++) {
      const b = list[ti];
      if (!b || b.pathType !== 'runway_exit' || b.id === rtxA.id) continue;
      if (rtxPolylinesTouch(rtxA, b)) ids.add(b.id);
    }
    return ids;
  }
  function expandRtxCandidateIdsTouchingLineup(runwayTw, lineupPt) {
    const hop1 = listRtxTouchingLineupOnRunway(runwayTw, lineupPt);
    const ids = new Set();
    hop1.forEach(function(tx) { if (tx && tx.id != null) ids.add(tx.id); });
    const list = state.taxiways || [];
    hop1.forEach(function(a) {
      for (let ti = 0; ti < list.length; ti++) {
        const b = list[ti];
        if (!b || b.pathType !== 'runway_exit' || b.id === a.id) continue;
        if (ids.has(b.id)) continue;
        if (rtxPolylinesTouch(a, b)) ids.add(b.id);
      }
    });
    return { hop1: hop1, allIds: ids };
  }
  function holdingPointWorldXY(hp) {
    if (!hp) return null;
    if (typeof hp.x === 'number' && isFinite(hp.x) && typeof hp.y === 'number' && isFinite(hp.y)) return [hp.x, hp.y];
    return null;
  }
  function runwayHoldingNearRtxCandidateSet(hp, candIds) {
    const p = holdingPointWorldXY(hp);
    if (!p || normalizeHoldingPointKind(hp.hpKind) !== 'runway_holding') return false;
    const tolD2 = lineupHoldingTolD2(1.15);
    const list = state.taxiways || [];
    for (let ti = 0; ti < list.length; ti++) {
      const tx = list[ti];
      if (tx.pathType !== 'runway_exit' || !candIds.has(tx.id)) continue;
      const rtxPts = getOrderedPoints(tx);
      if (rtxPts && rtxPts.length >= 2 && pointNearPolylineSq(p, rtxPts, tolD2)) return true;
    }
    return false;
  }
  function cumulativeDistAlongPolylineToPoint(pts, q) {
    if (!pts || pts.length < 2 || !q) return null;
    let best = null;
    let acc = 0;
    for (let i = 0; i < pts.length - 1; i++) {
      const a = pts[i], b = pts[i + 1];
      const segLen = pathDist(a, b);
      if (segLen < 1e-9) continue;
      const pr = projectOnSegment(a, b, q);
      const t = Math.max(0, Math.min(1, pr.t));
      const d = dist2(pr.p, q);
      const cand = { distAlong: acc + t * segLen, d2: d, proj: pr.p };
      if (!best || d < best.d2) best = cand;
      acc += segLen;
    }
    return best;
  }
  function findLastRunwayHoldingOnDeparturePath(toLineup, candIds) {
    if (!toLineup || toLineup.length < 2) return null;
    const hps = state.holdingPoints || [];
    let best = null;
    for (let hi = 0; hi < hps.length; hi++) {
      const hp = hps[hi];
      if (!hp || normalizeHoldingPointKind(hp.hpKind) !== 'runway_holding') continue;
      if (!runwayHoldingNearRtxCandidateSet(hp, candIds)) continue;
      const p = holdingPointWorldXY(hp);
      if (!p) continue;
      const tolD2 = lineupHoldingTolD2(1.3);
      if (!pointNearPolylineSq(p, toLineup, tolD2)) continue;
      const cum = cumulativeDistAlongPolylineToPoint(toLineup, p);
      if (!cum) continue;
      if (!best || cum.distAlong > best.distAlong) best = { hp: hp, distAlong: cum.distAlong, proj: cum.proj };
    }
    return best;
  }
  function polylineDurationSecTaxi(pts) {
    if (!pts || pts.length < 2) return 0;
    const carry = { lastTaxiwayMs: null };
    let sec = 0;
    for (let i = 0; i < pts.length - 1; i++) {
      const len = pathDist(pts[i], pts[i + 1]);
      if (len < 1e-9) continue;
      const v = taxiSegmentVelocityMsForPolylineSegment(pts[i], pts[i + 1], carry);
      sec += len / Math.max(0.1, v);
    }
    return sec;
  }
  function depTakeoffAccelMs2ForFlight(f) {
    const ac = typeof getAircraftInfoByType === 'function' ? getAircraftInfoByType(f && f.aircraftType) : null;
    const mtow = typeof ac.mtow_kg === 'number' && isFinite(ac.mtow_kg) ? ac.mtow_kg : 100000;
    if (mtow <= DEP_MTOW_REF_SMALL_KG) return DEP_TAKEOFF_ACCEL_SMALL_MS2;
    if (mtow >= DEP_MTOW_REF_LARGE_KG) return DEP_TAKEOFF_ACCEL_LARGE_MS2;
    const t = (mtow - DEP_MTOW_REF_SMALL_KG) / (DEP_MTOW_REF_LARGE_KG - DEP_MTOW_REF_SMALL_KG);
    return DEP_TAKEOFF_ACCEL_SMALL_MS2 + t * (DEP_TAKEOFF_ACCEL_LARGE_MS2 - DEP_TAKEOFF_ACCEL_SMALL_MS2);
  }
  function takeoffRollSecForRunwayTailLenM(lenM, accelMs2) {
    const L = Math.max(0, lenM);
    const a = Math.max(0.1, accelMs2);
    if (L < 1e-6) return 0;
    return Math.sqrt(2 * L / a);
  }
  function polylineTotalLenM(pts) {
    if (!pts || pts.length < 2) return 0;
    let s = 0;
    for (let i = 0; i < pts.length - 1; i++) s += pathDist(pts[i], pts[i + 1]);
    return s;
  }
  function rtxSetHasRunwayHoldingHp(candIds) {
    const hps = state.holdingPoints || [];
    for (let i = 0; i < hps.length; i++) {
      const hp = hps[i];
      if (!hp || normalizeHoldingPointKind(hp.hpKind) !== 'runway_holding') continue;
      if (runwayHoldingNearRtxCandidateSet(hp, candIds)) return true;
    }
    return false;
  }
  function computeDepHoldToLineupSecForFlight(f) {
    if (!f || f.arrDep !== 'Dep' || f.noWayDep) return 0;
    const toLineup = (typeof graphPathDeparture === 'function') ? graphPathDeparture(f, { onlyToLineup: true }) : null;
    if (!toLineup || toLineup.length < 2) return 0;
    const runwayId = f.depRunwayId || (f.token && (f.token.depRunwayId != null ? f.token.depRunwayId : f.token.runwayId)) || f.arrRunwayId;
    const rwTw = (state.taxiways || []).find(function(t) { return t && t.id === runwayId && t.pathType === 'runway'; });
    const lineupPt = toLineup[toLineup.length - 1];
    const exp = rwTw ? expandRtxCandidateIdsTouchingLineup(rwTw, lineupPt) : { allIds: new Set() };
    const holdPick = findLastRunwayHoldingOnDeparturePath(toLineup, exp.allIds);
    if (!holdPick || !(holdPick.distAlong > 1e-3)) return 0;
    const spl = polylineSplitAtDistance(toLineup, holdPick.distAlong);
    const holdToLineup = spl.second && spl.second.length >= 2 ? spl.second : null;
    if (!holdToLineup) return 0;
    return polylineDurationSecTaxi(holdToLineup);
  }
  function computeDepRollAndLineupOnlySec(f) {
    const split = (typeof splitDeparturePathLineupAndRunwayTail === 'function') ? splitDeparturePathLineupAndRunwayTail(f) : null;
    const a = depTakeoffAccelMs2ForFlight(f);
    const tailLen = split && split.runwayTail ? polylineTotalLenM(split.runwayTail) : 0;
    return DEP_LINEUP_HOLD_SEC + takeoffRollSecForRunwayTailLenM(tailLen, a);
  }
  function computeDepRotSecondsForFlight(f) {
    return computeDepHoldToLineupSecForFlight(f) + computeDepRollAndLineupOnlySec(f);
  }
  function dedupePathPoints(pts) {
    const out = [];
    (pts || []).forEach(function(p) {
      if (!p || p.length < 2) return;
      if (!out.length || dist2(out[out.length - 1], p) > SPLIT_TOL_D2) out.push([p[0], p[1]]);
    });
    return out;
  }
  function polylineDistanceBetweenAlong(pts, startAlong, endAlong) {
    if (!pts || pts.length < 2) return 0;
    const a0 = Math.max(0, Number(startAlong) || 0);
    const a1 = Math.max(a0, Number(endAlong) || 0);
    let dist = 0;
    for (let seg = Math.floor(a0); seg <= Math.min(pts.length - 2, Math.floor(a1)); seg++) {
      const segStart = Math.max(seg, a0);
      const segEnd = Math.min(seg + 1, a1);
      if (segEnd <= segStart) continue;
      const segLen = pathDist(pts[seg], pts[seg + 1]);
      if (!(segLen > 1e-9)) continue;
      dist += segLen * (segEnd - segStart);
    }
    return dist;
  }
  function polylinePointsBetweenAlong(pts, startAlong, endAlong) {
    if (!pts || pts.length < 2) return [];
    const a0 = Math.max(0, Number(startAlong) || 0);
    const a1 = Math.max(a0, Number(endAlong) || 0);
    const startSeg = Math.max(0, Math.min(pts.length - 2, Math.floor(a0)));
    const endSeg = Math.max(0, Math.min(pts.length - 2, Math.floor(a1)));
    const startT = a0 - startSeg;
    const endT = a1 - endSeg;
    const startPt = [
      pts[startSeg][0] + (pts[startSeg + 1][0] - pts[startSeg][0]) * startT,
      pts[startSeg][1] + (pts[startSeg + 1][1] - pts[startSeg][1]) * startT
    ];
    const endPt = [
      pts[endSeg][0] + (pts[endSeg + 1][0] - pts[endSeg][0]) * endT,
      pts[endSeg][1] + (pts[endSeg + 1][1] - pts[endSeg][1]) * endT
    ];
    const out = [[startPt[0], startPt[1]]];
    for (let i = startSeg + 1; i <= endSeg; i++) out.push([pts[i][0], pts[i][1]]);
    out.push([endPt[0], endPt[1]]);
    return dedupePathPoints(out);
  }
  /** p1 indices are on g1; p2 on gFull. buildPathFromIndices must use one graph — remap g1 nodes into gFull index space. */
  function retSplitPathIndicesOnGFull(g1, gFull, p1, p2, pivotIdx, pivotIdxFull) {
    if (!g1 || !gFull || !p1 || !p2 || p1.length < 2 || p2.length < 2) return null;
    const p1Seg = (pivotIdx === pivotIdxFull) ? p1 : p1.slice(0, -1);
    const part1 = [];
    for (let i = 0; i < p1Seg.length; i++) {
      const wp = g1.nodes[p1Seg[i]];
      if (!wp) return null;
      const ni = nearestPathNode(gFull, wp);
      if (!part1.length || part1[part1.length - 1] !== ni) part1.push(ni);
    }
    const p2Tail = (pivotIdx === pivotIdxFull) ? p2.slice(1) : p2;
    const merged = part1.concat(p2Tail);
    const out = [];
    for (let i = 0; i < merged.length; i++) {
      if (!out.length || out[out.length - 1] !== merged[i]) out.push(merged[i]);
    }
    return out.length >= 2 ? out : null;
  }

  function buildPathFromIndices(g, pathIndices) {
    if (!g || !Array.isArray(pathIndices) || pathIndices.length < 2) return null;
    const out = [];
    for (let i = 0; i < pathIndices.length - 1; i++) {
      const key = pathIndices[i] + ':' + pathIndices[i + 1];
      const edge = g.edgeMap ? g.edgeMap[key] : null;
      const pts = (edge && Array.isArray(edge.pts) && edge.pts.length >= 2)
        ? edge.pts
        : [g.nodes[pathIndices[i]], g.nodes[pathIndices[i + 1]]];
      pts.forEach(function(p) {
        if (!p || p.length < 2) return;
        if (!out.length || dist2(out[out.length - 1], p) > SPLIT_TOL_D2) out.push([p[0], p[1]]);
      });
    }
    return out;
  }

  /** F2: operational runway direction vs RET "Available RW direction" only (arrival sample — no extra geometry gate). */
  function arrivalRetPassesFilter2RunwayAvailableDir(rw, exitTw) {
    if (!rw || !exitTw || rw.pathType !== 'runway' || exitTw.pathType !== 'runway_exit') return false;
    const rd = getRunwayOperationalDirForArrivalRetFilter2(rw);
    if (rd === 'clockwise') return isRunwayExitDirAllowedForArrivalFilter2(exitTw, 'clockwise');
    return isRunwayExitDirAllowedForArrivalFilter2(exitTw, 'counter_clockwise');
  }

  function isPointOnRunwayPolyline2D(rVerts, q) {
    if (!rVerts || rVerts.length < 2 || !q) return false;
    for (let i = 0; i < rVerts.length - 1; i++) {
      if (pointOnSegmentStrict(rVerts[i], rVerts[i + 1], q)) return true;
    }
    return false;
  }

  /**
   * Arrival: if RET path direction (Runway exit mode) is not "both", require at least one edge
   * (fromIdx -> toIdx) to match: CCW — from off runway, to on; CW — from on, to off.
   */
  function arrivalRunwayExitPassPathDirEdgeToRunwayFilter(rw, exitTw, rVerts) {
    if (!exitTw || exitTw.pathType !== 'runway_exit' || !rVerts) return true;
    const pathDir = normalizeRwDirectionValue(getTaxiwayDirection(exitTw));
    if (pathDir === 'both') return true;
    const verts = exitTw.vertices;
    if (!Array.isArray(verts) || verts.length < 2) return false;
    for (let ei = 0; ei < verts.length - 1; ei++) {
      const f = [verts[ei].col, verts[ei].row];
      const t = [verts[ei + 1].col, verts[ei + 1].row];
      const onF = isPointOnRunwayPolyline2D(rVerts, f);
      const onT = isPointOnRunwayPolyline2D(rVerts, t);
      if (pathDir === 'counter_clockwise' && !onF && onT) return true;
      if (pathDir === 'clockwise' && onF && !onT) return true;
    }
    return false;
  }

  function computeRunwayExitDistances() {
    const taxiways = state.taxiways || [];
    const runways = taxiways.filter(t => t.pathType === 'runway' && Array.isArray(t.vertices) && t.vertices.length >= 2);
    const exits = taxiways.filter(t => t.pathType === 'runway_exit' && Array.isArray(t.vertices) && t.vertices.length >= 2);
    const results = [];
    if (!runways.length || !exits.length) return results;

    runways.forEach(rw => {
      let rVerts = rw.vertices.map(v => [v.col, v.row]);
      if (rVerts.length < 2) return;
      const prefixDist = [0];
      for (let i = 1; i < rVerts.length; i++) {
        prefixDist[i] = prefixDist[i - 1] + pathDist(rVerts[i - 1], rVerts[i]);
      }
      exits.forEach(tw => {
        let best = null;
        const exitName = (tw.name && tw.name.trim()) ? tw.name.trim() : ('Exit ' + String(results.length + 1));
        function considerRunwayHit(distCells) {
          const dCells = Math.max(0, distCells);
          const distM = dCells * CELL_SIZE;
          const maxExitVelRaw = (typeof tw.maxExitVelocity === 'number' && isFinite(tw.maxExitVelocity) && tw.maxExitVelocity > 0)
            ? tw.maxExitVelocity
            : 30;
          const minExitVelRaw = (typeof tw.minExitVelocity === 'number' && isFinite(tw.minExitVelocity) && tw.minExitVelocity > 0)
            ? tw.minExitVelocity
            : 15;
          const maxExitVel = maxExitVelRaw;
          const minExitVel = Math.min(minExitVelRaw, maxExitVel);
          if (!best || distM < best.distM) {
            best = { runway: rw, exit: tw, name: exitName, distM, maxExitVelocity: maxExitVel, minExitVelocity: minExitVel };
          }
        }
        tw.vertices.forEach(v => {
          const q = [v.col, v.row];
          for (let i = 0; i < rVerts.length - 1; i++) {
            const a = rVerts[i], b = rVerts[i + 1];
            if (!pointOnSegmentStrict(a, b, q)) continue;
            const segLen = pathDist(a, b);
            if (!(segLen > 1e-6)) continue;
            const proj = projectOnSegment(a, b, q);
            const t = Math.max(0, Math.min(1, segLen > 0 ? pathDist(a, proj.p) / segLen : 0));
            const distCells = prefixDist[i] + segLen * t;
            considerRunwayHit(distCells);
          }
        });
        let ev = tw.vertices.map(v => [v.col, v.row]);
        for (let ei = 0; ei < ev.length - 1; ei++) {
          const ea = ev[ei], eb = ev[ei + 1];
          for (let i = 0; i < rVerts.length - 1; i++) {
            const ra = rVerts[i], rb = rVerts[i + 1];
            const segLen = pathDist(ra, rb);
            if (!(segLen > 1e-6)) continue;
            function distFromRunwayPoint(q) {
              const proj = projectOnSegment(ra, rb, q);
              if (proj.t < -1e-9 || proj.t > 1 + 1e-9) return;
              if (dist2(proj.p, q) > SPLIT_TOL_D2 * 4) return;
              const t = Math.max(0, Math.min(1, segLen > 0 ? pathDist(ra, proj.p) / segLen : 0));
              considerRunwayHit(prefixDist[i] + segLen * t);
            }
            const isec = segmentSegmentIntersection(ea, eb, ra, rb);
            if (isec) distFromRunwayPoint(isec.p);
            const ovRw = collinearSegmentOverlapOnAB(ra, rb, ea, eb);
            if (ovRw) {
              const rax = ra[0], ray = ra[1], rbx = rb[0], rby = rb[1];
              const rdx = rbx - rax, rdy = rby - ray;
              distFromRunwayPoint([rax + ovRw.t0 * rdx, ray + ovRw.t0 * rdy]);
              distFromRunwayPoint([rax + ovRw.t1 * rdx, ray + ovRw.t1 * rdy]);
            }
          }
        }
        if (best) {
          if (!arrivalRetPassesFilter2RunwayAvailableDir(rw, tw)) best = null;
        }
        if (best && !arrivalRunwayExitPassPathDirEdgeToRunwayFilter(rw, tw, rVerts)) {
          best = null;
        }
        if (best) results.push(best);
      });
    });

    results.sort((a, b) => a.distM - b.distM);
    return results;
  }

  
  function mergeNearbyPathPointsForDraw(points, radiusM) {
    if (!points || !points.length) return [];
    const r = (typeof radiusM === 'number' && isFinite(radiusM) && radiusM > 0) ? radiusM : PATH_JUNCTION_MERGE_RADIUS_PX;
    const n = points.length;
    const parent = [];
    for (let i = 0; i < n; i++) parent[i] = i;
    function dsFind(i) {
      if (parent[i] !== i) parent[i] = dsFind(parent[i]);
      return parent[i];
    }
    function dsUnion(i, j) {
      const ri = dsFind(i), rj = dsFind(j);
      if (ri !== rj) parent[Math.max(ri, rj)] = Math.min(ri, rj);
    }
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        if (pathDist(points[i], points[j]) <= r) dsUnion(i, j);
      }
    }
    const buckets = {};
    for (let i = 0; i < n; i++) {
      const root = dsFind(i);
      if (!buckets[root]) buckets[root] = [];
      buckets[root].push(points[i]);
    }
    const out = [];
    Object.keys(buckets).forEach(function(k) {
      const g = buckets[k];
      let sx = 0, sy = 0;
      for (let t = 0; t < g.length; t++) { sx += g[t][0]; sy += g[t][1]; }
      out.push([sx / g.length, sy / g.length]);
    });
    return out;
  }

  
  function computeConnectedRunwayExitIds(seedId, pathList) {
    const out = new Set();
    if (seedId == null) return out;
    const rex = (pathList || []).filter(function(tw) {
      return tw && tw.pathType === 'runway_exit' && getOrderedPoints(tw) && getOrderedPoints(tw).length >= 2;
    });
    const idToTw = {};
    rex.forEach(function(tw) { idToTw[tw.id] = tw; });
    const touchD2 = Math.max(SPLIT_TOL_D2, Math.pow(CELL_SIZE * 0.2, 2));
    function twPairTouch(twA, twB) {
      const p1 = getOrderedPoints(twA);
      const p2 = getOrderedPoints(twB);
      if (!p1 || !p2 || p1.length < 2 || p2.length < 2) return false;
      let i, s, pr;
      for (i = 0; i < p1.length; i++) {
        for (s = 0; s < p2.length - 1; s++) {
          pr = projectOnSegment(p2[s], p2[s + 1], p1[i]);
          if (dist2(pr.p, p1[i]) <= touchD2) return true;
        }
      }
      for (i = 0; i < p2.length; i++) {
        for (s = 0; s < p1.length - 1; s++) {
          pr = projectOnSegment(p1[s], p1[s + 1], p2[i]);
          if (dist2(pr.p, p2[i]) <= touchD2) return true;
        }
      }
      return false;
    }
    if (!idToTw[seedId]) {
      out.add(seedId);
      return out;
    }
    const queue = [seedId];
    out.add(seedId);
    while (queue.length) {
      const curId = queue.shift();
      const curTw = idToTw[curId];
      if (!curTw) continue;
      rex.forEach(function(tw) {
        if (out.has(tw.id)) return;
        if (twPairTouch(tw, curTw)) {
          out.add(tw.id);
          queue.push(tw.id);
        }
      });
    }
    return out;
  }

  function queueTaxiwayAutoJunctionMarkersAlong(tw, spacingM) {
    const verts = tw && tw.vertices;
    if (!verts || verts.length < 2 || !isFinite(spacingM) || spacingM < 1e-6) return [];
    if (String(tw.pathType || '') !== 'general_queue_taxiway') return [];
    const out = [];
    let alongM = 0;
    let nextMark = spacingM;
    for (let i = 0; i < verts.length - 1; i++) {
      const v0 = verts[i], v1 = verts[i + 1];
      const dc = Number(v1.col) - Number(v0.col);
      const dr = Number(v1.row) - Number(v0.row);
      const segM = Math.hypot(dc, dr) * CELL_SIZE;
      if (segM < 1e-12) continue;
      while (alongM + segM >= nextMark - 1e-9) {
        const intoSegM = nextMark - alongM;
        const u = intoSegM / segM;
        const pu = Math.max(0, Math.min(1, u));
        const col = Number(v0.col) + pu * dc;
        const row = Number(v0.row) + pu * dr;
        const p = cellToPixel(col, row);
        out.push({ tAlong: i + pu, p });
        nextMark += spacingM;
      }
      alongM += segM;
    }
    return out;
  }

  function layoutWorldViewportAabbWorldM() {
    if (!layoutDrawCanvas) return { minWx: -Infinity, maxWx: Infinity, minWy: -Infinity, maxWy: Infinity, marginWorld: 0 };
    const w = layoutDrawCanvas.width / dpr;
    const h = layoutDrawCanvas.height / dpr;
    const s = state.scale || 1;
    const marginWorld = CELL_SIZE * Math.max(6, 96 / Math.max(s, 0.06));
    return {
      minWx: (0 - state.panX) / s - marginWorld,
      maxWx: (w - state.panX) / s + marginWorld,
      minWy: (0 - state.panY) / s - marginWorld,
      maxWy: (h - state.panY) / s + marginWorld,
      marginWorld: marginWorld
    };
  }
  function worldPointInsideLayoutViewportAabb(p, vb) {
    if (!p || p.length < 2 || !vb) return false;
    return p[0] >= vb.minWx && p[0] <= vb.maxWx && p[1] >= vb.minWy && p[1] <= vb.maxWy;
  }
  function taxiwayWorldAabb(tw) {
    if (!tw || !tw.vertices || !tw.vertices.length) return null;
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (let vi = 0; vi < tw.vertices.length; vi++) {
      const v = tw.vertices[vi];
      const xy = cellToPixel(Number(v.col), Number(v.row));
      minX = Math.min(minX, xy[0]); maxX = Math.max(maxX, xy[0]);
      minY = Math.min(minY, xy[1]); maxY = Math.max(maxY, xy[1]);
    }
    return { minX: minX, minY: minY, maxX: maxX, maxY: maxY };
  }
  function taxiwayWorldAabbIntersectsViewport(tw, vb) {
    const a = taxiwayWorldAabb(tw);
    if (!a || !vb) return true;
    const pad = CELL_SIZE * 3;
    return !(a.maxX + pad < vb.minWx - pad || a.minX - pad > vb.maxWx + pad || a.maxY + pad < vb.minWy - pad || a.minY - pad > vb.maxWy + pad);
  }
  function taxiwayShouldDrawInViewport(tw, vb) {
    if (!tw || !vb) return true;
    if (state.taxiwayDrawingId != null && String(state.taxiwayDrawingId) === String(tw.id)) return true;
    const so = state.selectedObject;
    if (so && so.type === 'taxiway' && String(so.id) === String(tw.id)) return true;
    return taxiwayWorldAabbIntersectsViewport(tw, vb);
  }
  function terminalWorldAabbFromVertices(term) {
    if (!term || !Array.isArray(term.vertices) || !term.vertices.length) return null;
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    let ok = false;
    for (let i = 0; i < term.vertices.length; i++) {
      const v = term.vertices[i];
      if (!v) continue;
      const col = Number(v.col), row = Number(v.row);
      if (!isFinite(col) || !isFinite(row)) continue;
      const x = col * CELL_SIZE;
      const y = row * CELL_SIZE;
      ok = true;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
    return ok ? { minX: minX, minY: minY, maxX: maxX, maxY: maxY } : null;
  }
  const LAYOUT_RENDER_VIEWPORT_BUFFER_M = 200;
  function layoutWorldViewportAabbWithBufferM(bufferM) {
    if (!layoutDrawCanvas) return { minWx: -Infinity, maxWx: Infinity, minWy: -Infinity, maxWy: Infinity, marginWorld: 0 };
    const w = layoutDrawCanvas.width / dpr;
    const h = layoutDrawCanvas.height / dpr;
    const s = state.scale || 1;
    const m = Math.max(0, Number(bufferM) || 0);
    return {
      minWx: (0 - state.panX) / s - m,
      maxWx: (w - state.panX) / s + m,
      minWy: (0 - state.panY) / s - m,
      maxWy: (h - state.panY) / s + m,
      marginWorld: m
    };
  }
  function aabbIntersectsViewport(vb, aabb) {
    if (!vb || !aabb) return true;
    return !(aabb.maxX < vb.minWx || aabb.minX > vb.maxWx || aabb.maxY < vb.minWy || aabb.minY > vb.maxWy);
  }
  function pointsWorldAabb(points) {
    if (!Array.isArray(points) || !points.length) return null;
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    let ok = false;
    for (let i = 0; i < points.length; i++) {
      const p = points[i];
      if (!p || p.length < 2) continue;
      const x = Number(p[0]), y = Number(p[1]);
      if (!isFinite(x) || !isFinite(y)) continue;
      ok = true;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
    return ok ? { minX: minX, minY: minY, maxX: maxX, maxY: maxY } : null;
  }
  function markerWorldAabb(m) {
    if (!m) return null;
    const pts = [];
    if (Array.isArray(m.points)) {
      for (let i = 0; i < m.points.length; i++) {
        const p = m.points[i];
        if (!p) continue;
        const x = Number(p.x), y = Number(p.y);
        if (isFinite(x) && isFinite(y)) pts.push([x, y]);
      }
    }
    [['x', 'y'], ['x1', 'y1'], ['x2', 'y2']].forEach(function(pair) {
      const x = Number(m[pair[0]]), y = Number(m[pair[1]]);
      if (isFinite(x) && isFinite(y)) pts.push([x, y]);
    });
    const a = pointsWorldAabb(pts);
    if (!a) return null;
    const pad = Math.max(8, CELL_SIZE * 0.8);
    return { minX: a.minX - pad, minY: a.minY - pad, maxX: a.maxX + pad, maxY: a.maxY + pad };
  }
  function overlayJunctionFillForWorldPoint(p, gCache) {
    if (layerMonoEtcOn()) return C2D_LAYER_MONO_ETC_WHITE;
    if (!gCache || !p) return '#22c55e';
