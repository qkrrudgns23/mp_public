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
  }
  function prepareLazyTimelinesForCurrentSim(tSec) {
    if (!state.globalUpdateFresh) return;
    const flights = state.flights || [];
    const pad = simAirsideLazyPadSec();
    for (let i = 0; i < flights.length; i++) {
      const f = flights[i];
      if (!f) continue;
      if (flightBlockedLikeNoWay(f)) continue;
      if (f.timeline_meta && f.timeline_meta.playbackSource === 'des_result') continue;
      if (!f.timeline || !f.timeline.length) continue;
      const w = getFlightAirsideWindowSec(f);
      if (!w) { f.timeline = null; continue; }
      if (tSec > w.t1 + 1e-3 || tSec < w.t0 - pad - 1e-3) f.timeline = null;
    }
    const pending = [];
    for (let i = 0; i < flights.length; i++) {
      const f = flights[i];
      if (!f) continue;
      if (flightBlockedLikeNoWay(f)) continue;
      if (f.timeline_meta && f.timeline_meta.playbackSource === 'des_result') continue;
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

  
  function getEffectiveRunwayLineupDistM(tw) {
    if (!tw || tw.pathType !== 'runway') return 0;
    const v = tw.lineupDistM;
    if (typeof v === 'number' && isFinite(v) && v >= 0) return v;
    return 0;
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
    const dStart = Math.min(Math.max(0, getEffectiveRunwayStartDisplacedThresholdM(tw) + getEffectiveRunwayStartBlastPadM(tw)), totalLen);
    const endInset = getEffectiveRunwayEndDisplacedThresholdM(tw) + getEffectiveRunwayEndBlastPadM(tw);
    const dEnd = Math.max(0, Math.min(totalLen, totalLen - endInset));
    if (!(totalLen > 1e-6)) return dStart;
    if (tdDistAlong <= totalLen * 0.5) return dStart;
    return dEnd;
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

  function drawRunwayDecorations(tw, pts, widthPx) {
    if (!tw || tw.pathType !== 'runway' || !tw.start_point || !tw.end_point) return;
    if (!pts || pts.length < 2) return;
    const totalLen = runwayPolylineLengthPx(pts);
    const runwayWidth = Math.max(24, Number(widthPx) || RUNWAY_PATH_DEFAULT_WIDTH);
    if (totalLen < Math.max(220, runwayWidth * 3)) return;
    const startDisp = getEffectiveRunwayStartDisplacedThresholdM(tw);
    const startBlast = getEffectiveRunwayStartBlastPadM(tw);
    const endDisp = getEffectiveRunwayEndDisplacedThresholdM(tw);
    const endBlast = getEffectiveRunwayEndBlastPadM(tw);
    const startFrame = getPolylinePointAndFrameAtDistance(pts, 0);
    const endFrame = getPolylinePointAndFrameAtDistance(pts, totalLen);
    if (!startFrame || !endFrame) return;

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
    const thresholdColor = c2dRunwayThresholdColor();
    const touchdownColor = c2dRunwayTouchdownColor();
    const aimingPointColor = c2dRunwayAimingPointColor();
    const extensionFill = c2dRunwayExtensionFill();
    const extensionOutline = c2dRunwayOutline();
    const blastChevronColor = c2dRunwayBlastChevronColor();

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
      ctx.fillStyle = thresholdColor;
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

    drawExtensionSegment(startFrame, -1, 0, startDisp);
    drawExtensionSegment(startFrame, -1, startDisp, startBlast);
    drawExtensionSegment(endFrame, 1, 0, endDisp);
    drawExtensionSegment(endFrame, 1, endDisp, endBlast);
    drawDisplacedThresholdArrows(startFrame, -1, 1, 0, startDisp);
    drawDisplacedThresholdArrows(endFrame, 1, -1, 0, endDisp);
    drawBlastPadChevrons(startFrame, -1, startDisp, startBlast);
    drawBlastPadChevrons(endFrame, 1, endDisp, endBlast);

    const thresholdInset = Math.min(Math.max(runwayWidth * 0.58, 26), totalLen * 0.12);
    const thresholdStripeLen = Math.min(Math.max(runwayWidth * 0.54, 20), 34);
    const thresholdStripeWidth = Math.max(3, runwayWidth * 0.085);
    [-runwayWidth * 0.30, -runwayWidth * 0.18, -runwayWidth * 0.06, runwayWidth * 0.06, runwayWidth * 0.18, runwayWidth * 0.30].forEach(function(offset) {
      drawRectAtBothEnds(thresholdInset, offset, thresholdStripeLen, thresholdStripeWidth, thresholdColor);
    });

    (function drawRunwayCenterlineDashed() {
      const paveStart = startDisp + startBlast;
      const paveEnd = totalLen - endDisp - endBlast;
      if (!(paveEnd > paveStart + 1)) return;
      const clPts = polylineSliceBetweenDistances(pts, paveStart, paveEnd);
      if (!clPts || clPts.length < 2) return;
      ctx.save();
      ctx.strokeStyle = c2dRunwayCenterlineColor();
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
    })();

    const aimingDist = Math.min(Math.max(300, runwayWidth * 3.5), totalLen * 0.28);
    if (aimingDist < (totalLen * 0.5) - (runwayWidth * 0.6)) {
      drawSymmetricPairAtBothEnds(
        aimingDist,
        runwayWidth * 0.20,
        Math.min(Math.max(runwayWidth * 1.2, 54), 92),
        Math.max(6, runwayWidth * 0.12),
        aimingPointColor
      );
    }

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
  function getTaxiwayOrderedPoints(tw) {
    if (!tw.vertices || tw.vertices.length < 2) return null;
    const pts = tw.vertices.map(v => cellToPixel(v.col, v.row));
    if (tw.start_point && tw.end_point) {
      const startPx = cellToPixel(tw.start_point.col, tw.start_point.row);
      if (dist2(pts[pts.length-1], startPx) < dist2(pts[0], startPx)) pts.reverse();
    }
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
  
  function isLineupPointTouchingRunwayTaxiwayOnRunway(runwayTw, lineupPt) {
    if (!runwayTw || runwayTw.pathType !== 'runway' || !lineupPt) return false;
    const rwPts = getOrderedPoints(runwayTw);
    if (!rwPts || rwPts.length < 2) return false;
    const cs = (typeof CELL_SIZE === 'number' && isFinite(CELL_SIZE) && CELL_SIZE > 0) ? CELL_SIZE : 20;
    const touchD2 = Math.max(SPLIT_TOL_D2, (cs * 0.2) * (cs * 0.2));
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
    const cs = (typeof CELL_SIZE === 'number' && isFinite(CELL_SIZE) && CELL_SIZE > 0) ? CELL_SIZE : 20;
    const touchD2 = Math.max(SPLIT_TOL_D2, (cs * 0.2) * (cs * 0.2));
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
    const cs = (typeof CELL_SIZE === 'number' && isFinite(CELL_SIZE) && CELL_SIZE > 0) ? CELL_SIZE : 20;
    const tolD2 = Math.max(SPLIT_TOL_D2, (cs * 0.35) * (cs * 0.35));
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
      const cs = (typeof CELL_SIZE === 'number' && isFinite(CELL_SIZE) && CELL_SIZE > 0) ? CELL_SIZE : 20;
      const tolD2 = Math.max(SPLIT_TOL_D2, (cs * 0.45) * (cs * 0.45));
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

  function computeRunwayExitDistances() {
    const taxiways = state.taxiways || [];
    const runways = taxiways.filter(t => t.pathType === 'runway' && Array.isArray(t.vertices) && t.vertices.length >= 2);
    const exits = taxiways.filter(t => t.pathType === 'runway_exit' && Array.isArray(t.vertices) && t.vertices.length >= 2);
    const results = [];
    if (!runways.length || !exits.length) return results;

    runways.forEach(rw => {
      let rVerts = rw.vertices.map(v => [v.col, v.row]);
      if (rw.start_point && rw.end_point && rVerts.length >= 2) {
        const sp = [rw.start_point.col, rw.start_point.row];
        if (dist2(rVerts[rVerts.length - 1], sp) < dist2(rVerts[0], sp)) rVerts.reverse();
      }
      if (rVerts.length < 2) return;
      const prefixDist = [0];
      for (let i = 1; i < rVerts.length; i++) {
        prefixDist[i] = prefixDist[i - 1] + pathDist(rVerts[i - 1], rVerts[i]);
      }

      exits.forEach(tw => {
        let best = null;
        const exitName = (tw.name && tw.name.trim()) ? tw.name.trim() : ('Exit ' + String(results.length + 1));
        function considerRunwayHit(distCells) {
          const distM = distCells * CELL_SIZE;
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
        if (tw.start_point && tw.end_point && ev.length >= 2) {
          const sp = [tw.start_point.col, tw.start_point.row];
          if (dist2(ev[ev.length - 1], sp) < dist2(ev[0], sp)) ev.reverse();
        }
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
          const rwOpDir = normalizeRwDirectionValue(getTaxiwayDirection(rw));
          if ((rwOpDir === 'clockwise' || rwOpDir === 'counter_clockwise') &&
              !isRunwayExitDirectionAllowed(tw, rwOpDir)) {
            best = null;
          }
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

  function buildPathGraph(selectedArrRetId, runwayDirectionForExit, pathGraphOpts) {
    const opts = pathGraphOpts && typeof pathGraphOpts === 'object' ? pathGraphOpts : {};
    const pureGroundExcludeRunway = !!opts.pureGroundExcludeRunway;
    /** When true with selectedArrRetId: drop other runway_exit polylines (runway→chosen RET leg only). Full taxi/crossing needs all RETs. */
    const omitOtherRunwayExits = !!opts.omitOtherRunwayExits;
    const nodes = [], keyToIdx = {}, edges = [], adj = [], junctionPts = [], junctionKeys = {}, edgeMap = {};
    const nodeBucket = {};
    const mergeRM = PATH_JUNCTION_MERGE_RADIUS_PX;
    function nodeBucketKeyForPoint(p) {
      return Math.floor(p[0] / mergeRM) + ',' + Math.floor(p[1] / mergeRM);
    }
    function findNodeIndexWithinMergeRadius(p) {
      const bx = Math.floor(p[0] / mergeRM);
      const by = Math.floor(p[1] / mergeRM);
      for (let dx = -1; dx <= 1; dx++) {
        for (let dy = -1; dy <= 1; dy++) {
          const list = nodeBucket[(bx + dx) + ',' + (by + dy)];
          if (!list) continue;
          for (let t = 0; t < list.length; t++) {
            const idx = list[t];
            if (pathDist(nodes[idx], p) <= mergeRM) return idx;
          }
        }
      }
      return null;
    }
    const runwayNodeIndicesById = {};
    function addJunction(p) {
      const k = pathPointKey(p);
      if (junctionKeys[k]) return;
      junctionKeys[k] = true;
      junctionPts.push(p);
    }
    function getOrAdd(p) {
      const found = findNodeIndexWithinMergeRadius(p);
      if (found != null) return found;
      const idx = nodes.length;
      nodes.push([p[0], p[1]]);
      const k = pathPointKey(p);
      keyToIdx[k] = idx;
      adj[idx] = [];
      const bkey = nodeBucketKeyForPoint(p);
      if (!nodeBucket[bkey]) nodeBucket[bkey] = [];
      nodeBucket[bkey].push(idx);
      return idx;
    }
    function registerDirectedEdge(fromIdx, toIdx, cost, rawDist, pts, linkId, pathType, pathDir) {
      const edge = {
        from: fromIdx,
        to: toIdx,
        dist: cost,
        rawDist: rawDist,
        pts: dedupePathPoints(pts),
        linkId: linkId != null ? String(linkId) : '',
        pathType: pathType != null ? String(pathType) : 'taxiway',
        pathDir: pathDir != null ? String(pathDir) : 'both'
      };
      edges.push(edge);
      edgeMap[fromIdx + ':' + toIdx] = edge;
    }
    function addEdgeWithDirection(pFrom, pTo, dir, cost, rawDist, ptsForward, linkId, pathType) {
      const i = getOrAdd(pFrom), j = getOrAdd(pTo);
      if (i === j || cost < 1e-6) return;
      const forwardPts = dedupePathPoints(ptsForward && ptsForward.length ? ptsForward : [pFrom, pTo]);
      const reversePts = forwardPts.slice().reverse();
      const lid = linkId != null ? String(linkId) : '';
      const pt = pathType != null ? String(pathType) : 'taxiway';
      registerDirectedEdge(i, j, cost, rawDist, forwardPts, lid, pt, dir);
      if (dir === 'both') {
        adj[i].push([j, cost]);
        adj[j].push([i, cost]);
        registerDirectedEdge(j, i, cost, rawDist, reversePts, lid, pt, dir);
      } else if (dir === 'counter_clockwise') {
        adj[j].push([i, cost]);
        adj[i].push([j, REVERSE_COST]);
        registerDirectedEdge(i, j, REVERSE_COST, rawDist, forwardPts, lid, pt, dir);
      } else {
        adj[i].push([j, cost]);
        adj[j].push([i, REVERSE_COST]);
        registerDirectedEdge(j, i, REVERSE_COST, rawDist, reversePts, lid, pt, dir);
      }
    }

    const pathList = state.taxiways || [];
    const apronNodeStand = [];
    const minD2 = 1e-6;
    pathList.forEach(obj => {
      if (omitOtherRunwayExits && selectedArrRetId != null && obj && obj.pathType === 'runway_exit' && obj.id !== selectedArrRetId) return;
      const pts = getOrderedPoints(obj);
      if (!pts || pts.length < 2) return;
      const junctions = [];
      for (let seg = 0; seg < pts.length - 1; seg++) {
        const a = pts[seg], b = pts[seg+1];
        pathList.forEach(other => {
          if (other.id === obj.id) return;
          const otherOrd = getOrderedPoints(other);
          if (!otherOrd || otherOrd.length < 2) return;
          for (let oseg = 0; oseg < otherOrd.length - 1; oseg++) {
            const c = otherOrd[oseg], d = otherOrd[oseg+1];
            const isec = segmentSegmentIntersection(a, b, c, d);
            if (isec) {
              const { t } = projectOnSegment(a, b, isec.p);
              junctions.push({ tAlong: seg + t, p: isec.p });
            } else {
              const ov = collinearSegmentOverlapOnAB(a, b, c, d);
              if (ov) {
                const ax = a[0], ay = a[1], bx = b[0], by = b[1];
                const dx = bx - ax, dy = by - ay;
                const p0 = [ax + ov.t0 * dx, ay + ov.t0 * dy];
                const p1ov = [ax + ov.t1 * dx, ay + ov.t1 * dy];
                const pr0 = projectOnSegment(a, b, p0);
                junctions.push({ tAlong: seg + pr0.t, p: pr0.p });
                if (dist2(p0, p1ov) > SPLIT_TOL_D2) {
                  const pr1 = projectOnSegment(a, b, p1ov);
                  junctions.push({ tAlong: seg + pr1.t, p: pr1.p });
                }
              } else {
              [c, d].forEach(function(q, idx) {
                if (dist2(a, q) <= SPLIT_TOL_D2 || dist2(b, q) <= SPLIT_TOL_D2) {
                  const { t, p: proj } = projectOnSegment(a, b, q);
                  if (t >= 0 && t <= 1) junctions.push({ tAlong: seg + t, p: proj });
                }
              });
              }
            }
          }
          otherOrd.forEach(q => {
            if (!pointOnSegmentStrict(a, b, q)) return;
            const { t, p: proj } = projectOnSegment(a, b, q);
            junctions.push({ tAlong: seg + t, p: proj });
          });
        });
        const isRunway = obj.pathType === 'runway';
        if (!isRunway) {
          (state.apronLinks || []).forEach(lk => {
            if (lk.taxiwayId !== obj.id || lk.tx == null || lk.ty == null) return;
            const linkPt = [Number(lk.tx), Number(lk.ty)];
            const { t, p } = projectOnSegment(a, b, linkPt);
            if (t >= 0 && t <= 1 && dist2(p, linkPt) <= SPLIT_TOL_D2) {
              junctions.push({ tAlong: seg + t, p });
              const pbb = findStandById(lk.pbbId);
              if (pbb) {
                const standPt = getStandConnectionPx(pbb);
                const mids = (Array.isArray(lk.midVertices) ? lk.midVertices : []).map(function(v) { return cellToPixel(Number(v.col), Number(v.row)); });
                const chain = [standPt].concat(mids).concat([p]);
                apronNodeStand.push({ nodeP: p, standPt, standId: lk.pbbId, chain, linkId: lk.id || 'apron_link' });
              }
            }
          });
        }
      }
      if (obj.pathType === 'runway') {
        const ldm = getEffectiveRunwayLineupDistM(obj);
        const rpath = getRunwayPath(obj.id);
        if (rpath && rpath.pts && rpath.pts.length >= 2 && ldm > 1e-6) {
          let total = 0;
          for (let ri = 0; ri < rpath.pts.length - 1; ri++) total += pathDist(rpath.pts[ri], rpath.pts[ri + 1]);
          const d = Math.min(ldm, total);
          if (d > 1e-6) {
            let acc = 0;
            for (let ri = 0; ri < rpath.pts.length - 1; ri++) {
              const p1 = rpath.pts[ri], p2 = rpath.pts[ri + 1];
              const segLen = pathDist(p1, p2);
              if (segLen < 1e-9) continue;
              if (acc + segLen >= d - 1e-6) {
                const t = Math.max(0, Math.min(1, (d - acc) / segLen));
                const px = p1[0] + t * (p2[0] - p1[0]), py = p1[1] + t * (p2[1] - p1[1]);
                junctions.push({ tAlong: ri + t, p: [px, py] });
                break;
              }
              acc += segLen;
            }
          }
        }
      }
      const waypoints = [
        { tAlong: 0, p: pts[0], isJunction: false },
        { tAlong: pts.length - 1, p: pts[pts.length - 1], isJunction: false }
      ];
      junctions.forEach(({ tAlong, p }) => waypoints.push({ tAlong, p, isJunction: true }));
      waypoints.sort((x, y) => x.tAlong - y.tAlong);
      const chain = [];
      waypoints.forEach(function(wp) {
        if (chain.length && Math.abs(wp.tAlong - chain[chain.length - 1].tAlong) < 1e-9 && dist2(wp.p, chain[chain.length - 1].p) < minD2) {
          if (wp.isJunction) addJunction(wp.p);
          return;
        }
        chain.push({ tAlong: wp.tAlong, p: wp.p, isJunction: !!wp.isJunction });
        if (wp.isJunction) addJunction(wp.p);
      });
      if (obj.pathType === 'runway') {
        const runwayNodeSet = runwayNodeIndicesById[obj.id] || (runwayNodeIndicesById[obj.id] = new Set());
        chain.forEach(function(wp) {
          runwayNodeSet.add(getOrAdd(wp.p));
        });
      }
      const dir = getTaxiwayDirection(obj);
      const tw_id = String(obj.id || '');
      const path_type = String(obj.pathType || 'taxiway');
      const isRunwayExit = obj.pathType === 'runway_exit';
      const isTaxiway = obj.pathType === 'taxiway';
      for (let i = 0; i < chain.length - 1; i++) {
        const segPts = polylinePointsBetweenAlong(pts, chain[i].tAlong, chain[i + 1].tAlong);
        let d = polylineDistanceBetweenAlong(pts, chain[i].tAlong, chain[i + 1].tAlong);
        let cost = d;
        if (isRunwayExit && !isRunwayExitDirectionAllowed(obj, runwayDirectionForExit)) {
          cost = REVERSE_COST;
        }
        if (selectedArrRetId != null && isTaxiway) {
          cost = d + TAXIWAY_HEURISTIC_COST;
        }
        if (pureGroundExcludeRunway && obj.pathType === 'runway') cost = REVERSE_COST;
        addEdgeWithDirection(chain[i].p, chain[i + 1].p, dir, cost, d, segPts, tw_id, path_type);
      }
    });

    const standNodeIndices = [];
    const standIdToNodeIndex = {};
    apronNodeStand.forEach(function(entry) {
      const nodeP = entry.nodeP, standPt = entry.standPt, standId = entry.standId, chain = entry.chain;
      const apronLinkId = entry.linkId != null ? String(entry.linkId) : 'apron_link';
      const i = getOrAdd(nodeP);
      const j = getOrAdd(standPt);
      standNodeIndices.push(j);
      if (standId != null) standIdToNodeIndex[standId] = j;
      const pts = (chain && chain.length >= 2) ? dedupePathPoints(chain) : [nodeP, standPt];
      if (!pts || pts.length < 2 || i === j) return;
      let totalDist = 0;
      for (let k = 0; k < pts.length - 1; k++) totalDist += pathDist(pts[k], pts[k + 1]);
      if (!(totalDist > 1e-6)) return;
      adj[i].push([j, totalDist]);
      adj[j].push([i, totalDist]);
      registerDirectedEdge(i, j, totalDist, totalDist, pts.slice().reverse(), apronLinkId, 'apron_link', 'both');
      registerDirectedEdge(j, i, totalDist, totalDist, pts, apronLinkId, 'apron_link', 'both');
    });
    function bfsReachable(startIndices) {
      const out = new Set();
      const q = startIndices.slice();
      startIndices.forEach(function(idx) { out.add(idx); });
      while (q.length) {
        const u = q.shift();
        (adj[u] || []).forEach(function(tuple) {
          const v = tuple[0], w = tuple[1];
          if (w >= REVERSE_COST) return;
          if (!out.has(v)) { out.add(v); q.push(v); }
        });
      }
      return out;
    }
    function nearestNode(p) {
      let best = 0, bestD2 = dist2(nodes[0], p);
      for (let i = 1; i < nodes.length; i++) {
        const d2 = dist2(nodes[i], p);
        if (d2 < bestD2) { bestD2 = d2; best = i; }
      }
      return best;
    }
    const runwayNodeIndices = [];
    const runwayNodeSeen = new Set();
    const runways = (state.taxiways || []).filter(function(t) { return t.pathType === 'runway'; });
    runways.forEach(function(rw) {
      const r = getRunwayPath(rw.id);
      if (!r) return;
      [r.startPx, r.endPx].forEach(function(p) {
        if (!p) return;
        const idx = nearestNode(p);
        if (idx == null || runwayNodeSeen.has(idx)) return;
        runwayNodeSeen.add(idx);
        runwayNodeIndices.push(idx);
      });
    });
    const runwayReachable = runwayNodeIndices.length ? bfsReachable(runwayNodeIndices) : new Set();
    const standReachable = standNodeIndices.length ? bfsReachable(standNodeIndices) : new Set();
    const connected = new Set();
    runwayReachable.forEach(function(i) { if (standReachable.has(i)) connected.add(i); });
    const validJunctionsForDraw = junctionPts.filter(function(p) {
      const i = findNodeIndexWithinMergeRadius(p);
      return i != null && adj[i] && adj[i].length >= 2;
    });
    const connectedJunctionsForDraw = validJunctionsForDraw.filter(function(p) {
      const i = findNodeIndexWithinMergeRadius(p);
      return i != null && connected.has(i);
    });
    const disconnectedValidJunctionsForDraw = validJunctionsForDraw.filter(function(p) {
      const i = findNodeIndexWithinMergeRadius(p);
      return i != null && !connected.has(i);
    });
    const connectedJunctionsMerged = mergeNearbyPathPointsForDraw(connectedJunctionsForDraw, PATH_JUNCTION_MERGE_RADIUS_PX);
    return {
      nodes,
      edges,
      adj,
      edgeMap,
      getOrAdd,
      runwayNodeIndicesById,
      junctions: connectedJunctionsMerged,
      validJunctions: validJunctionsForDraw,
      disconnectedValidJunctions: disconnectedValidJunctionsForDraw,
      connectedJunctions: connectedJunctionsMerged,
      standIdToNodeIndex
    };
  }

  function serializePathGraphForSim(g) {
    if (!g || !g.nodes || !g.edges) return null;
    const runwayById = {};
    Object.keys(g.runwayNodeIndicesById || {}).forEach(function(k) {
      const setv = g.runwayNodeIndicesById[k];
      runwayById[k] = setv ? Array.from(setv) : [];
    });
    const standMap = {};
    Object.keys(g.standIdToNodeIndex || {}).forEach(function(k) {
      standMap[String(k)] = g.standIdToNodeIndex[k];
    });
    return {
      nodes: g.nodes.map(function(p) { return [+p[0], +p[1]]; }),
      edges: g.edges.map(function(e) {
        return {
          from: e.from,
          to: e.to,
          dist: e.dist,
          rawDist: e.rawDist != null ? e.rawDist : e.dist,
          pts: (e.pts || []).map(function(p) { return [+p[0], +p[1]]; }),
          linkId: e.linkId != null ? String(e.linkId) : '',
          pathType: e.pathType != null ? String(e.pathType) : 'taxiway',
          pathDir: e.pathDir != null ? String(e.pathDir) : 'both'
        };
      }),
      standIdToNodeIndex: standMap,
      runwayNodeIndicesById: runwayById
    };
  }

  function buildSimPathGraphExport() {
    if (!state.taxiways || !state.taxiways.length) return null;
    try {
      return {
        version: 1,
        reverseCost: REVERSE_COST,
        mergeRadiusPx: PATH_JUNCTION_MERGE_RADIUS_PX,
        clockwise: {
          standard: serializePathGraphForSim(buildPathGraph(null, 'clockwise')),
          pureGroundExcludeRunway: serializePathGraphForSim(
            buildPathGraph(null, 'clockwise', { pureGroundExcludeRunway: true })
          )
        },
        counter_clockwise: {
          standard: serializePathGraphForSim(buildPathGraph(null, 'counter_clockwise')),
          pureGroundExcludeRunway: serializePathGraphForSim(
            buildPathGraph(null, 'counter_clockwise', { pureGroundExcludeRunway: true })
          )
        }
      };
    } catch (err) {
      console.error('buildSimPathGraphExport failed', err);
      return null;
    }
  }

  function rebuildDerivedGraphEdges() {
    state.derivedGraphEdges = [];
    if (!state.taxiways || !state.taxiways.length) return;
    let g;
    try {
      g = buildPathGraph(null);
    } catch (err) {
      console.error('rebuildDerivedGraphEdges: buildPathGraph failed', err);
      return;
    }
    if (!g || !g.edges || !g.nodes) return;
    const seen = new Set();
    const raw = [];
    g.edges.forEach(function(e) {
      if (e.dist >= REVERSE_COST || e.dist < 1e-6) return;
      const a = e.from, b = e.to;
      const lo = a < b ? a : b, hi = a < b ? b : a;
      const k = lo + ':' + hi;
      if (seen.has(k)) return;
      seen.add(k);
      const p0 = g.nodes[a], p1 = g.nodes[b];
      if (!p0 || !p1) return;
      raw.push({
        x1: p0[0], y1: p0[1], x2: p1[0], y2: p1[1],
        pts: Array.isArray(e.pts) ? e.pts.map(function(p) { return [p[0], p[1]]; }) : [[p0[0], p0[1]], [p1[0], p1[1]]],
        dist: e.rawDist != null ? e.rawDist : e.dist,
        fromIdx: a, toIdx: b
      });
    });
    raw.sort(function(u, v) {
      if (u.fromIdx !== v.fromIdx) return u.fromIdx - v.fromIdx;
      return u.toIdx - v.toIdx;
    });
    const maxN = Math.min(raw.length, 999);
    const nextEdgeNames = {};
    const usedEdgeNames = new Set();
    for (let i = 0; i < maxN; i++) {
      const label = String(i + 1).padStart(3, '0');
      const r = raw[i];
      const edgeId = 'layout-edge-' + label;
      const preferredName = (state.layoutEdgeNames && state.layoutEdgeNames[edgeId]) || ('Edge ' + label);
      const finalName = uniqueNameAgainstSet(preferredName, usedEdgeNames);
      usedEdgeNames.add(finalName);
      nextEdgeNames[edgeId] = finalName;
      state.derivedGraphEdges.push({
        id: edgeId,
        label: label,
        name: finalName,
        x1: r.x1, y1: r.y1, x2: r.x2, y2: r.y2,
        pts: r.pts,
        dist: r.dist,
        fromIdx: r.fromIdx,
        toIdx: r.toIdx
      });
    }
    state.layoutEdgeNames = nextEdgeNames;
    if (state.selectedObject && state.selectedObject.type === 'layoutEdge') {
      const sid = state.selectedObject.id;
      const fresh = (state.derivedGraphEdges || []).find(function(e) { return e.id === sid; });
      if (fresh) state.selectedObject.obj = fresh;
      else state.selectedObject = null;
    }
  }

  function hitTestLayoutGraphEdge(wx, wy) {
    if (!state.derivedGraphEdges || !state.derivedGraphEdges.length) return null;
    const click = [wx, wy];
    const tol = CELL_SIZE * 0.4;
    const tol2 = tol * tol;
    let best = null, bestD2 = tol2;
    state.derivedGraphEdges.forEach(function(ed) {
      const pts = (ed.pts && ed.pts.length >= 2) ? ed.pts : [[ed.x1, ed.y1], [ed.x2, ed.y2]];
      for (let i = 0; i < pts.length - 1; i++) {
        const near = closestPointOnSegment(pts[i], pts[i + 1], click);
        if (!near) continue;
        const d2 = dist2(near, click);
        if (d2 < bestD2) { bestD2 = d2; best = ed; }
      }
    });
    return best;
  }

  class MinHeap {
    constructor() { this.h = []; }
    push(item) {
      this.h.push(item);
      let i = this.h.length - 1;
      while (i > 0) {
        const p = (i - 1) >> 1;
        if (this.h[p][0] <= this.h[i][0]) break;
        [this.h[p], this.h[i]] = [this.h[i], this.h[p]];
        i = p;
      }
    }
    pop() {
      const top = this.h[0];
      const last = this.h.pop();
      if (this.h.length) {
        this.h[0] = last;
        let i = 0;
        while (true) {
          let s = i, l = 2*i+1, r = 2*i+2;
          if (l < this.h.length && this.h[l][0] < this.h[s][0]) s = l;
          if (r < this.h.length && this.h[r][0] < this.h[s][0]) s = r;
          if (s === i) break;
          [this.h[s], this.h[i]] = [this.h[i], this.h[s]];
          i = s;
        }
      }
      return top;
    }
    get size() { return this.h.length; }
  }

  function pathDijkstra(g, startIdx, endIdx) {
    const n = g.nodes.length;
    const dist = Array(n).fill(Infinity);
    const prev = Array(n).fill(null);
    if (startIdx == null || endIdx == null) return null;
    dist[startIdx] = 0;
    const heap = new MinHeap();
    heap.push([0, startIdx]);
    while (heap.size) {
      const [d, u] = heap.pop();
      if (d > dist[u]) continue;
      if (u === endIdx) break;
      for (const [v, w] of g.adj[u]) {
        const nd = d + w;
        if (nd < dist[v]) {
          dist[v] = nd;
          prev[v] = u;
          heap.push([nd, v]);
        }
      }
    }
    if (dist[endIdx] === Infinity || dist[endIdx] >= REVERSE_COST) return null;
    const path = [];
    for (let cur = endIdx; cur !== null; cur = prev[cur]) path.push(cur);
    return path.reverse();
  }

  /** RET 출구 근처 여러 그래프 노드에서 gFull 상 스탠드까지 다익스트라를 시도해, 단일 nearest 스냅이 다른 성분에 묶이는 경우를 완화한다. */
  function gatherRetExitPivotIndicesOnGFull(gFull, retEndPx, pivotG1Px, rPts) {
    const mergeRM = PATH_JUNCTION_MERGE_RADIUS_PX;
    const pxPts = [];
    if (pivotG1Px && pivotG1Px.length >= 2) pxPts.push(pivotG1Px);
    if (retEndPx && retEndPx.length >= 2) pxPts.push(retEndPx);
    if (rPts && rPts.length >= 2) {
      pxPts.push(rPts[rPts.length - 1]);
      if (rPts.length >= 3) pxPts.push(rPts[rPts.length - 2]);
    }
    const indices = [];
    const seen = new Set();
    for (let i = 0; i < pxPts.length; i++) {
      const idx = nearestPathNode(gFull, pxPts[i]);
      if (idx != null && !seen.has(idx)) {
        seen.add(idx);
        indices.push(idx);
      }
    }
    const rNear = mergeRM * 6;
    const r2 = rNear * rNear;
    if (retEndPx && retEndPx.length >= 2 && gFull.nodes && gFull.nodes.length) {
      const scored = [];
      for (let ni = 0; ni < gFull.nodes.length; ni++) {
        const d2 = dist2(gFull.nodes[ni], retEndPx);
        if (d2 <= r2) scored.push({ ni: ni, d2: d2 });
      }
      scored.sort(function(a, b) { return a.d2 - b.d2; });
      const cap = 36;
      for (let k = 0; k < scored.length && k < cap; k++) {
        const ni = scored[k].ni;
        if (!seen.has(ni)) {
          seen.add(ni);
          indices.push(ni);
        }
      }
    }
    return indices;
  }
  function pathDijkstraFromRetExitToStand(gFull, endNodeFull, candidateStartIndices) {
    if (!gFull || endNodeFull == null || !candidateStartIndices || !candidateStartIndices.length) return { path: null, startIdx: null };
    let bestPath = null;
    let bestD = Infinity;
    const seenStart = new Set();
    for (let ci = 0; ci < candidateStartIndices.length; ci++) {
      const s = candidateStartIndices[ci];
      if (s == null || seenStart.has(s)) continue;
      seenStart.add(s);
      const path = pathDijkstra(gFull, s, endNodeFull);
      if (!path || path.length < 2) continue;
      const d = pathTotalDist(gFull, path);
      if (!(d < REVERSE_COST)) continue;
      if (d < bestD) {
        bestD = d;
        bestPath = path;
      }
    }
    return { path: bestPath, startIdx: bestPath ? bestPath[0] : null };
  }

  function nearestPathNode(g, p) {
    let best = 0, bestD2 = dist2(g.nodes[0], p);
    for (let i = 1; i < g.nodes.length; i++) {
      const d2 = dist2(g.nodes[i], p);
      if (d2 < bestD2) { bestD2 = d2; best = i; }
    }
    return best;
  }
  function nearestPathNodeFromSet(g, nodeSet, p) {
    if (!g || !g.nodes || !g.nodes.length || !nodeSet || !nodeSet.size) return null;
    let best = null, bestD2 = Infinity;
    nodeSet.forEach(function(idx) {
      if (idx == null || !g.nodes[idx]) return;
      const d2 = dist2(g.nodes[idx], p);
      if (d2 < bestD2) { bestD2 = d2; best = idx; }
    });
    return best;
  }
  /** Avoid snapping to another runway's polyline when multiple runways exist (same idea as departure lineup). */
  function nearestPathNodeOnRunwayPolyline(g, runwayId, runwayPx) {
    if (!g || !g.nodes || !g.nodes.length || !runwayPx) return null;
    const rwSet = g.runwayNodeIndicesById && g.runwayNodeIndicesById[runwayId];
    if (rwSet && rwSet.size)
      return nearestPathNodeFromSet(g, rwSet, runwayPx) ?? nearestPathNode(g, runwayPx);
    return nearestPathNode(g, runwayPx);
  }

  function pathTotalDist(g, pathIndices) {
    let d = 0;
    for (let i = 0; i < pathIndices.length - 1; i++) {
      const a = g.nodes[pathIndices[i]], b = g.nodes[pathIndices[i+1]];
      const e = g.edgeMap ? g.edgeMap[pathIndices[i] + ':' + pathIndices[i+1]] : null;
      if (e) d += e.dist; else d += pathDist(a, b);
    }
    return d;
  }

  function probePreferredArrivalRunwayDir(f) {
    const token = f.token || {};
    let runwayId = resolveArrivalRunwayIdForFlight(f);
    const apronId = f.standId != null ? f.standId : (token.apronId || null);
    if (!apronId || runwayId == null || runwayId === '') return 'both';
    const r = getRunwayPath(runwayId);
    if (!r || !findStandById(apronId)) return 'both';
    function solveByRunwayDir(rwDir) {
      const runwayPx = rwDir === 'counter_clockwise' ? r.endPx : r.startPx;
      const excludeRunwayGround = DEFAULT_ALLOW_RUNWAY_IN_GROUND_SEGMENT !== true;
      const gFull = buildPathGraph(null, rwDir, { pureGroundExcludeRunway: excludeRunwayGround });
      const endNodeFull = gFull.standIdToNodeIndex && gFull.standIdToNodeIndex[apronId];
      if (endNodeFull == null) return { chosen: null };
      const g = buildPathGraph(null, rwDir);
      const endNode = (g.standIdToNodeIndex && g.standIdToNodeIndex[apronId] != null) ? g.standIdToNodeIndex[apronId] : null;
      if (endNode == null) return { chosen: null };
      const startNode = nearestPathNodeOnRunwayPolyline(g, runwayId, runwayPx);
      const p = pathDijkstra(g, startNode, endNode);
      if (!p || p.length < 2) return { chosen: null };
      const d = pathTotalDist(g, p);
      if (!(d < REVERSE_COST)) return { chosen: null };
      return { chosen: { totalD: d, runwayDir: rwDir } };
    }
    const candCw = solveByRunwayDir('clockwise').chosen;
    const candCcw = solveByRunwayDir('counter_clockwise').chosen;
    let chosen = candCw;
    if (candCcw && (!candCw || candCcw.totalD < candCw.totalD)) chosen = candCcw;
    if (!chosen || !chosen.runwayDir) return 'both';
    return chosen.runwayDir;
  }
  function resolveArrivalRunwayDirForRetGate(f) {
    const fromFlight = normalizeRwDirectionValue(f.arrRunwayDirUsed);
    if (fromFlight === 'clockwise' || fromFlight === 'counter_clockwise') return fromFlight;
    const probed = probePreferredArrivalRunwayDir(f);
    if (probed === 'clockwise' || probed === 'counter_clockwise') return probed;
    return 'both';
  }

  function graphPathArrival(f) {
    if (f && f.deferPathCompute) return null;
    f._noWayArrDetail = '';
    const token = f.token || {};
    let runwayId = resolveArrivalRunwayIdForFlight(f);
    const apronId = f.standId != null ? f.standId : (token.apronId || null);
    if (!apronId) {
      f.noWayArr = true;
      f._noWayArrDetail = '게이트(스탠드)가 없습니다. 스케줄·배정 또는 기종·터미널 조건으로 주기장을 찾지 못했을 수 있습니다.';
      return null;
    }
    if (!runwayId && state.taxiways && state.taxiways.length) {
      const runways = state.taxiways.filter(t => t.pathType === 'runway' && t.vertices && t.vertices.length >= 2);
      if (runways.length) runwayId = runways[Math.floor(Math.random() * runways.length)].id;
    }
    if (!runwayId) {
      f.noWayArr = true;
      f._noWayArrDetail = '도착 활주로가 지정되지 않았고 레이아웃에 활주로 정의도 없습니다.';
      return null;
    }
    const r = getRunwayPath(runwayId);
    if (!r) {
      f.noWayArr = true;
      f._noWayArrDetail = '도착 활주로 폴리라인을 불러오지 못했습니다.';
      return null;
    }
    const stand = findStandById(apronId);
    if (!stand) {
      f.noWayArr = true;
      f._noWayArrDetail = '스탠드 ID가 레이아웃에 없습니다.';
      return null;
    }
    const scheduleArrRetTrim = (f.scheduleArrRetId != null && String(f.scheduleArrRetId).trim()) ? String(f.scheduleArrRetId).trim() : '';
    let selectedArrRetId = f.sampledArrRet != null ? f.sampledArrRet : null;
    if (scheduleArrRetTrim && (state.taxiways || []).some(function(t) { return t && t.id === scheduleArrRetTrim && t.pathType === 'runway_exit'; }))
      selectedArrRetId = scheduleArrRetTrim;
    const validSelectedArrRetId = (selectedArrRetId != null && (state.taxiways || []).some(function(t) {
      return t && t.id === selectedArrRetId && t.pathType === 'runway_exit';
    })) ? selectedArrRetId : null;
    if (selectedArrRetId != null && validSelectedArrRetId == null) {
      f.sampledArrRet = null;
      f.arrRetFailed = false;
      f.arrRotSec = null;
    }
    function solveByRunwayDir(rwDir) {
      const dirTag = rwDir === 'clockwise' ? '(시계) ' : '(반시계) ';
      const runwayPx = rwDir === 'counter_clockwise' ? r.endPx : r.startPx;
      const splitDiag = validSelectedArrRetId != null ? { rwDir: rwDir, retId: validSelectedArrRetId } : null;
      const excludeRunwayGround = DEFAULT_ALLOW_RUNWAY_IN_GROUND_SEGMENT !== true;
      const gFullOpts = { pureGroundExcludeRunway: excludeRunwayGround };
      const gFull = buildPathGraph(null, rwDir, gFullOpts);
      const endNodeFull = gFull.standIdToNodeIndex && gFull.standIdToNodeIndex[apronId];
      if (endNodeFull == null) {
        return {
          chosen: null,
          hint: dirTag + '순전 지상 그래프에 스탠드가 연결되어 있지 않습니다. 에이프런 링크·택시웨이를 확인하세요. (활주로 중심선을 순전 지상에 쓰려면 Information.json의 flight_schedule.defaultAllowRunwayInGroundSegment를 true로 두세요.)',
          splitDiag: splitDiag
        };
      }
      if (validSelectedArrRetId != null) {
        const retTw = (state.taxiways || []).find(function(t) {
          return t && t.id === validSelectedArrRetId && t.pathType === 'runway_exit';
        });
        const rPts = retTw ? getOrderedPoints(retTw) : null;
        if (!(rPts && rPts.length >= 2)) {
          if (splitDiag) splitDiag.retSplitFail = 'ret_polyline_short_or_missing';
        } else {
          const retEndPx = rPts[rPts.length - 1];
          const g1 = buildPathGraph(validSelectedArrRetId, rwDir, { omitOtherRunwayExits: true });
          const startNode = nearestPathNodeOnRunwayPolyline(g1, runwayId, runwayPx);
          const pivotIdx = nearestPathNode(g1, retEndPx);
          const pivotPxG1 = g1.nodes[pivotIdx] || retEndPx;
          const p1 = pathDijkstra(g1, startNode, pivotIdx);
          let p2 = null;
          let pivotIdxFull = null;
          if (p1 && p1.length) {
            const candIdx = gatherRetExitPivotIndicesOnGFull(gFull, retEndPx, pivotPxG1, rPts);
            const leg2 = pathDijkstraFromRetExitToStand(gFull, endNodeFull, candIdx);
            p2 = leg2.path;
            pivotIdxFull = leg2.startIdx;
            if (splitDiag) splitDiag.p2CandidateCount = candIdx.length;
          }
          if (splitDiag) {
            splitDiag.startNode = startNode;
            splitDiag.pivotIdx = pivotIdx;
            splitDiag.pivotIdxFull = pivotIdxFull;
            splitDiag.p1Len = p1 ? p1.length : 0;
            splitDiag.p2Len = p2 ? p2.length : 0;
          }
          if (p1 && p1.length >= 2 && p2 && p2.length >= 2) {
            const merged = retSplitPathIndicesOnGFull(g1, gFull, p1, p2, pivotIdx, pivotIdxFull);
            const d = pathTotalDist(g1, p1) + pathTotalDist(gFull, p2);
            if (splitDiag) {
              splitDiag.mergedLen = merged ? merged.length : 0;
              splitDiag.splitTotalD = d;
            }
            if (merged && merged.length >= 2 && d < REVERSE_COST) {
              if (splitDiag) splitDiag.retSplitOk = true;
              return { chosen: { g: gFull, pathIndices: merged, totalD: d, runwayDir: rwDir, usedRetSplit: true }, hint: '', splitDiag: splitDiag };
            }
            if (splitDiag) {
              if (!merged || merged.length < 2) splitDiag.retSplitFail = 'merge_short_or_null';
              else if (!(d < REVERSE_COST)) splitDiag.retSplitFail = 'split_cost_reverse_or_limit';
            }
          } else if (splitDiag) {
            splitDiag.retSplitFail = (!p1 || p1.length < 2) ? 'p1_too_short' : 'p2_too_short_or_null';
          }
        }
      }
      const g = buildPathGraph(validSelectedArrRetId, rwDir);
      const endNode = (g.standIdToNodeIndex && g.standIdToNodeIndex[apronId] != null) ? g.standIdToNodeIndex[apronId] : null;
      if (endNode == null) {
        return {
          chosen: null,
          hint: dirTag + '경로 그래프에 스탠드 노드가 없습니다.',
          splitDiag: splitDiag
        };
      }
      const startNode = nearestPathNodeOnRunwayPolyline(g, runwayId, runwayPx);
      const p = pathDijkstra(g, startNode, endNode);
      if (!p || p.length < 2) {
        return {
          chosen: null,
          hint: dirTag + '활주로 쪽에서 스탠드까지 그래프가 끊어져 있습니다.',
          splitDiag: splitDiag
        };
      }
      const d = pathTotalDist(g, p);
      if (!(d < REVERSE_COST)) {
        return {
          chosen: null,
          hint: dirTag + '택시 경로 비용이 허용 한도를 초과했습니다(역방향·금지 구간).',
          splitDiag: splitDiag
        };
      }
      if (splitDiag) splitDiag.fellThroughToDirect = true;
      return { chosen: { g: g, pathIndices: p, totalD: d, runwayDir: rwDir, usedRetSplit: false }, hint: '', splitDiag: splitDiag };
    }
    const tryCw = solveByRunwayDir('clockwise');
    const tryCcw = solveByRunwayDir('counter_clockwise');
    const candCw = tryCw.chosen;
    const candCcw = tryCcw.chosen;
    let chosen = candCw;
    if (validSelectedArrRetId != null) {
      const splitCw = !!(candCw && candCw.usedRetSplit);
      const splitCcw = !!(candCcw && candCcw.usedRetSplit);
      if (splitCw && !splitCcw) chosen = candCw;
      else if (splitCcw && !splitCw) chosen = candCcw;
      else if (candCcw && (!candCw || candCcw.totalD < candCw.totalD)) chosen = candCcw;
    } else if (candCcw && (!candCw || candCcw.totalD < candCw.totalD)) {
      chosen = candCcw;
    }
    if (!chosen) {
      f.noWayArr = true;
      const hints = [tryCw.hint, tryCcw.hint].filter(function(h) { return h && String(h).trim(); });
      f._noWayArrDetail = hints.length ? Array.from(new Set(hints)).join(' ') : '시계·반시계 모두 도착 택시 경로를 찾지 못했습니다.';
      return null;
    }
    f.noWayArr = false;
    f._noWayArrDetail = '';
    state.pathGraphJunctions = chosen.g.junctions || [];
    f.arrRunwayDirUsed = chosen.runwayDir;
    return buildPathFromIndices(chosen.g, chosen.pathIndices);
  }

  function graphPathDeparture(f, opts) {
    if (f && f.deferPathCompute) return null;
    f._noWayDepDetail = '';
    opts = opts || {};
    const onlyToLineup = !!opts.onlyToLineup;
    const token = f.token || {};
    let runwayId = token.depRunwayId || token.runwayId || f.depRunwayId || f.arrRunwayId;
    const apronId = f.standId != null ? f.standId : (token.apronId || null);
    if (!apronId) {
      f.noWayDep = true;
      f._noWayDepDetail = '게이트(스탠드)가 없어 출발 경로를 시작할 수 없습니다.';
      return null;
    }
    if (!runwayId && state.taxiways && state.taxiways.length) {
      const runways = state.taxiways.filter(t => t.pathType === 'runway' && t.vertices && t.vertices.length >= 2);
      if (runways.length) runwayId = runways[Math.floor(Math.random() * runways.length)].id;
    }
    if (!runwayId) {
      f.noWayDep = true;
      f._noWayDepDetail = '출발 활주로가 없습니다.';
      return null;
    }
    const r = getRunwayPath(runwayId);
    if (!r) {
      f.noWayDep = true;
      f._noWayDepDetail = '출발 활주로 폴리라인을 불러오지 못했습니다.';
      return null;
    }
    const rwTw = (state.taxiways || []).find(t => t.id === runwayId && t.pathType === 'runway');
    const stand = findStandById(apronId);
    if (!stand) {
      f.noWayDep = true;
      f._noWayDepDetail = '스탠드 ID가 레이아웃에 없습니다.';
      return null;
    }
    const useLineup = rwTw && rwTw.pathType === 'runway';
    const runwayDirPref = normalizeRwDirectionValue(getTaxiwayDirection(rwTw));
    function solveDepartureByRunwayDir(rwDir) {
      const dirTag = rwDir === 'clockwise' ? '(시계) ' : '(반시계) ';
      const excludeRg = DEFAULT_ALLOW_RUNWAY_IN_GROUND_SEGMENT !== true;
      const g = buildPathGraph(null, rwDir, { pureGroundExcludeRunway: excludeRg });
      const startIdx = (g.standIdToNodeIndex && g.standIdToNodeIndex[apronId] != null) ? g.standIdToNodeIndex[apronId] : null;
      if (startIdx == null) {
        return {
          chosen: null,
          hint: dirTag + '순전 지상 그래프에 스탠드가 없습니다. 에이프런 링크를 확인하세요. (필요 시 Information.json flight_schedule.defaultAllowRunwayInGroundSegment.)'
        };
      }
      const useReverse = rwDir === 'counter_clockwise';
      const rPts = useReverse ? r.pts.slice().reverse() : r.pts.slice();
      const rStart = rPts[0];
      const rEnd = rPts[rPts.length - 1];
      if (useLineup) {
        const ldm = getEffectiveRunwayLineupDistM(rwTw);
        const lenPx = runwayPolylineLengthPx(rPts);
        const dPx = Math.min(Math.max(0, ldm), lenPx);
        const lineupFrame = getPolylinePointAndFrameAtDistance(rPts, dPx);
        const lineupPx = lineupFrame ? lineupFrame.point : null;
        if (!lineupPx) {
          return {
            chosen: null,
            hint: dirTag + '활주로 라인업 지점을 계산하지 못했습니다.'
          };
        }
        const runwayNodeSet = new Set(
          g.runwayNodeIndicesById && g.runwayNodeIndicesById[runwayId]
            ? Array.from(g.runwayNodeIndicesById[runwayId])
            : []
        );
        const lineupIdx = nearestPathNodeFromSet(g, runwayNodeSet, lineupPx) ?? nearestPathNode(g, lineupPx);
        const pathIndices = pathDijkstra(g, startIdx, lineupIdx);
        const totalD = pathIndices ? pathTotalDist(g, pathIndices) : Infinity;
        if (!pathIndices || pathIndices.length < 2 || totalD >= REVERSE_COST) {
          return {
            chosen: null,
            hint: dirTag + '스탠드에서 라인업 지점까지 경로가 없거나 금지 비용 한도를 초과했습니다.'
          };
        }
        let pts = buildPathFromIndices(g, pathIndices);
        if (!pts || pts.length < 2) {
          return {
            chosen: null,
            hint: dirTag + '라인업 구간 폴리라인을 만들지 못했습니다.'
          };
        }
        if (onlyToLineup) {
          return { chosen: { pts: pts, runwayDir: rwDir, totalD: totalD, g: g }, hint: '' };
        }
        const tail = polylineTailFromDistancePx(rPts, dPx);
        if (tail.length) {
          const last = pts[pts.length - 1];
          const firstTail = tail[0];
          if (dist2(last, firstTail) <= SPLIT_TOL_D2) pts = pts.concat(tail.slice(1));
          else pts = pts.concat(tail);
        }
        if (rEnd && Array.isArray(rEnd) && rEnd.length === 2) {
          const last = pts[pts.length - 1];
          if (pathDist(last, rEnd) > 1e-3) pts.push([rEnd[0], rEnd[1]]);
        }
        return { chosen: { pts: pts, runwayDir: rwDir, totalD: totalD, g: g }, hint: '' };
      }
      const runwayTargetIdx = nearestPathNode(g, rStart);
      const pathIndices = pathDijkstra(g, startIdx, runwayTargetIdx);
      const totalD = pathIndices ? pathTotalDist(g, pathIndices) : Infinity;
      if (!pathIndices || pathIndices.length < 2 || totalD >= REVERSE_COST) {
        return {
          chosen: null,
          hint: dirTag + '스탠드에서 활주로 쪽 접점까지 경로가 없거나 금지 비용을 초과했습니다.'
        };
      }
      const pts = buildPathFromIndices(g, pathIndices);
      if (!pts || pts.length < 2) {
        return {
          chosen: null,
          hint: dirTag + '출발 택시 폴리라인을 만들지 못했습니다.'
        };
      }
      return { chosen: { pts: pts, runwayDir: rwDir, totalD: totalD, g: g }, hint: '' };
    }
    let chosen = null;
    let depFailHints = [];
    if (runwayDirPref === 'clockwise' || runwayDirPref === 'counter_clockwise') {
      const one = solveDepartureByRunwayDir(runwayDirPref);
      chosen = one.chosen;
      if (!chosen && one.hint) depFailHints.push(one.hint);
    } else {
      const tryCw = solveDepartureByRunwayDir('clockwise');
      const tryCcw = solveDepartureByRunwayDir('counter_clockwise');
      const candCw = tryCw.chosen;
      const candCcw = tryCcw.chosen;
      chosen = candCw;
      if (candCcw && (!candCw || candCcw.totalD < candCw.totalD)) chosen = candCcw;
      if (!chosen) {
        if (tryCw.hint) depFailHints.push(tryCw.hint);
        if (tryCcw.hint) depFailHints.push(tryCcw.hint);
      }
    }
    if (!chosen) {
      f.noWayDep = true;
      const uniq = Array.from(new Set(depFailHints.filter(function(h) { return h && String(h).trim(); })));
      f._noWayDepDetail = uniq.length ? uniq.join(' ') : '출발 택시 경로를 찾지 못했습니다.';
      return null;
    }
    f.noWayDep = false;
    f._noWayDepDetail = '';
    f.depRunwayDirUsed = chosen.runwayDir;
    return chosen.pts;
  }

  function clonePathPtsForCache(pts) {
    if (!Array.isArray(pts) || pts.length < 2) return null;
    const out = [];
    for (let i = 0; i < pts.length; i++) {
      const p = pts[i];
      if (Array.isArray(p) && p.length >= 2) out.push([Number(p[0]), Number(p[1])]);
    }
    return out.length >= 2 ? out : null;
  }

  
  function normalizedArrRetCacheKey(f) {
    const id = f.sampledArrRet != null ? f.sampledArrRet : null;
    if (id == null) return '';
    const ok = (state.taxiways || []).some(function(t) {
      return t && t.id === id && t.pathType === 'runway_exit';
    });
    return ok ? String(id) : '';
  }

  function getPathForFlight(f) {
    if (f && f.deferPathCompute) return null;
    resolveStand(f);
    const arrRetKey = normalizedArrRetCacheKey(f);
    if (
      f._pathPolylineCacheRev === state.pathPolylineCacheRev &&
      String(f._pathPolylineArrRetKey || '') === arrRetKey &&
      Array.isArray(f.cachedArrPathPts) &&
      f.cachedArrPathPts.length >= 2 &&
      !arrivalAirsideBlocked(f)
    ) {
      return f.cachedArrPathPts;
    }
    const pts = graphPathArrival(f);
    if (pts && pts.length >= 2 && !arrivalAirsideBlocked(f)) {
      const cloned = clonePathPtsForCache(pts);
      if (cloned) {
        f.cachedArrPathPts = cloned;
        f._pathPolylineCacheRev = state.pathPolylineCacheRev;
        f._pathPolylineArrRetKey = normalizedArrRetCacheKey(f);
      }
    } else {
      delete f.cachedArrPathPts;
      delete f._pathPolylineArrRetKey;
    }
    return pts;
  }

  function getPathForFlightDeparture(f) {
    if (f && f.deferPathCompute) return null;
    resolveStand(f);
    if (
      f._pathPolylineCacheRev === state.pathPolylineCacheRev &&
      Array.isArray(f.cachedDepPathPts) &&
      f.cachedDepPathPts.length >= 2 &&
      !f.noWayDep
    ) {
      return f.cachedDepPathPts;
    }
    const pts = graphPathDeparture(f);
    if (pts && pts.length >= 2 && !f.noWayDep) {
      const cloned = clonePathPtsForCache(pts);
      if (cloned) {
        f.cachedDepPathPts = cloned;
        f._pathPolylineCacheRev = state.pathPolylineCacheRev;
      }
    } else {
      delete f.cachedDepPathPts;
    }
    return pts;
  }

  function ensureFlightPaths(f) {
    if (f && f.deferPathCompute) return;
    getPathForFlight(f);
    getPathForFlightDeparture(f);
    if (flightBlockedLikeNoWay(f)) f.timeline = null;
  }

  function findStandById(id) {
    return (state.pbbStands || []).find(function(s) { return s.id === id; }) ||
           (state.remoteStands || []).find(function(s) { return s.id === id; });
  }

  function buildTimeAxisTicks(minT, maxT, baseMinT, baseSpan, zoom) {
    const span = maxT - minT;
    const axisStep = span <= 60 ? TICK_STEP_SPAN_LE60 : (span <= 240 ? TICK_STEP_SPAN_LE240 : TICK_STEP_ELSE);
    let ticks = [];
    let tt = Math.floor(minT / axisStep) * axisStep;
    while (tt + 1e-9 < minT) tt += axisStep;
    while (tt <= maxT + 1e-9) {
      const leftPct = baseSpan > 1e-9 ? ((tt - baseMinT) / baseSpan) * 100 * zoom : 0;
      ticks.push({ leftPct: leftPct, label: formatMinutesToHHMM(tt) });
      tt += axisStep;
    }
    if (ticks.length > MAX_TICKS_SHOWN) {
      const step = Math.ceil(ticks.length / MAX_TICKS_SHOWN);
      const reduced = [];
      for (let i = 0; i < ticks.length; i += step) reduced.push(ticks[i]);
      const last = ticks[ticks.length - 1];
      if (reduced[reduced.length - 1] !== last) reduced.push(last);
      ticks = reduced;
    }
    return ticks;
  }

  function computeFlightPath(flight, direction) {
    if (flight && flight.deferPathCompute) {
      return { pts: null, timeline: null };
    }
    resolveStand(flight);
    if (direction === 'arrival') {
      const pts = graphPathArrival(flight);
      if (pts && pts.length >= 2 && !arrivalAirsideBlocked(flight)) {
        const cloned = clonePathPtsForCache(pts);
        if (cloned) {
          flight.cachedArrPathPts = cloned;
          flight._pathPolylineCacheRev = state.pathPolylineCacheRev;
          flight._pathPolylineArrRetKey = normalizedArrRetCacheKey(flight);
        }
      } else {
        delete flight.cachedArrPathPts;
        delete flight._pathPolylineArrRetKey;
      }
      return { pts: pts || null, timeline: null };
    }
    const pts = graphPathDeparture(flight);
    if (pts && pts.length >= 2 && !flight.noWayDep) {
      const cloned = clonePathPtsForCache(pts);
      if (cloned) {
        flight.cachedDepPathPts = cloned;
        flight._pathPolylineCacheRev = state.pathPolylineCacheRev;
      }
    } else {
      delete flight.cachedDepPathPts;
    }
    return { pts: pts || null, timeline: null };
  }

  const FLIGHT_PATH_PROGRESS_PCT_START = 22;
  const FLIGHT_PATH_PROGRESS_PCT_END = 48;
  const PATH_DIRECTION_ARROWS_MAX = 48;
  function updateAllFlightPaths(onDone) {
    if (!state.flights || !state.flights.length) {
      draw();
      if (typeof onDone === 'function') onDone();
      return;
    }
    const flights = state.flights;
    const asyncDone = typeof onDone === 'function';
    function applyPathsForFlight(f) {
      computeFlightPath(f, 'arrival');
      computeFlightPath(f, 'departure');
      if (flightBlockedLikeNoWay(f)) f.timeline = null;
    }
    function finishPaths() {
      if (typeof clearAllFlightTimelines === 'function') clearAllFlightTimelines();
      if (typeof syncSimulationPlaybackAfterTimelines === 'function') syncSimulationPlaybackAfterTimelines();
      if (typeof renderFlightList === 'function') renderFlightList(true);
      draw();
      if (asyncDone) onDone();
    }
    if (!asyncDone) {
      flights.forEach(applyPathsForFlight);
      finishPaths();
      return;
    }
    const totalFlights = flights.length;
    let i = 0;
    function pathChunk() {
      if (i >= totalFlights) {
        finishPaths();
        return;
      }
      applyPathsForFlight(flights[i]);
      i++;
      if (typeof setGlobalUpdateProgressUi === 'function') {
        const span = FLIGHT_PATH_PROGRESS_PCT_END - FLIGHT_PATH_PROGRESS_PCT_START;
        const pct = totalFlights > 0
          ? FLIGHT_PATH_PROGRESS_PCT_START + Math.round(span * (i / totalFlights))
          : FLIGHT_PATH_PROGRESS_PCT_START;
        setGlobalUpdateProgressUi(true, '항공 경로 ' + i + '/' + totalFlights, pct);
      }
      if (i < totalFlights) setTimeout(pathChunk, 0);
      else finishPaths();
    }
    setTimeout(pathChunk, 0);
  }

  function drawPathJunctions() {
    let g = null;
    if (state.taxiways && state.taxiways.length) {
      try { g = buildPathGraph(); } catch (e) { console.error('drawPathJunctions: buildPathGraph failed', e); }
    }
    if (!g) return;
    const validJunctions = g.validJunctions || [];
    const connectedJunctions = g.connectedJunctions || g.junctions || [];
    const redJunctions = g.disconnectedValidJunctions != null ? g.disconnectedValidJunctions : validJunctions;
    if (!validJunctions.length && !connectedJunctions.length) return;
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.translate(state.panX, state.panY);
    ctx.scale(state.scale, state.scale);
    const r = Math.max(4, CELL_SIZE * 0.35) * LAYOUT_VERTEX_DOT_SCALE;
    ctx.fillStyle = '#ef4444';
    redJunctions.forEach(p => {
      ctx.beginPath();
      ctx.arc(p[0], p[1], r, 0, Math.PI * 2);
      ctx.fill();
    });
    ctx.fillStyle = '#22c55e';
    connectedJunctions.forEach(p => {
      ctx.beginPath();
      ctx.arc(p[0], p[1], r, 0, Math.PI * 2);
      ctx.fill();
    });
    ctx.fillStyle = '#0f172a';
    ctx.font = (Math.max(7, CELL_SIZE * 0.18)) + 'px system-ui';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    (g.edges || []).forEach(e => {
      if (e.dist >= REVERSE_COST || e.dist < 1e-6) return;
      const a = g.nodes[e.from], b = g.nodes[e.to];
      if (!a || !b) return;
      const mx = (a[0] + b[0]) / 2, my = (a[1] + b[1]) / 2;
      ctx.fillText(Math.round(e.dist).toString(), mx, my);
    });
    ctx.restore();
  }

  function drawSelectedLayoutEdge() {
    const sel = state.selectedObject;
    if (!sel || sel.type !== 'layoutEdge' || !sel.obj) return;
    const e = sel.obj;
    const edgePts = (e.pts && e.pts.length >= 2) ? e.pts : [[e.x1, e.y1], [e.x2, e.y2]];
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.translate(state.panX, state.panY);
    ctx.scale(state.scale, state.scale);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    function layoutEdgePath() {
      ctx.beginPath();
      ctx.moveTo(edgePts[0][0], edgePts[0][1]);
      for (let i = 1; i < edgePts.length; i++) ctx.lineTo(edgePts[i][0], edgePts[i][1]);
    }
    layoutEdgePath();
    ctx.save();
    ctx.setLineDash([]);
    ctx.lineWidth = Math.max(7, CELL_SIZE * 0.2);
    ctx.strokeStyle = c2dObjectSelectedStroke();
    ctx.shadowColor = c2dObjectSelectedGlow();
    ctx.shadowBlur = c2dObjectSelectedGlowBlur();
    ctx.shadowOffsetX = 0;
    ctx.shadowOffsetY = 0;
    ctx.stroke();
    ctx.restore();
    layoutEdgePath();
    ctx.setLineDash([]);
    ctx.lineWidth = Math.max(4, CELL_SIZE * 0.12);
    ctx.strokeStyle = c2dObjectSelectedStroke();
    ctx.stroke();
    ctx.restore();
  }

  function drawProSimFlightPathEdges() {
    const sel = state.selectedObject;
    const rid = state.flightPathRevealFlightId;
    if (!sel || sel.type !== 'flight' || !sel.obj || rid == null || String(sel.id) !== String(rid)) return;
    const ids = sel.obj.edge_list || sel.obj.proSimEdgeList;
    if (!Array.isArray(ids) || !ids.length) return;
    if (typeof rebuildDerivedGraphEdges === 'function') rebuildDerivedGraphEdges();
    const byId = {};
    (state.derivedGraphEdges || []).forEach(function(ed) {
      if (ed && ed.id) byId[ed.id] = ed;
    });
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.translate(state.panX, state.panY);
    ctx.scale(state.scale, state.scale);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    const glow = '#4ade80';
    const core = '#f7fee7';
    ids.forEach(function(eid) {
      const key = eid != null ? String(eid).trim() : '';
      const e = key ? byId[key] : null;
      if (!e) return;
      const edgePts = (e.pts && e.pts.length >= 2) ? e.pts : [[e.x1, e.y1], [e.x2, e.y2]];
      ctx.beginPath();
      ctx.moveTo(edgePts[0][0], edgePts[0][1]);
      for (let i = 1; i < edgePts.length; i++) ctx.lineTo(edgePts[i][0], edgePts[i][1]);
      ctx.save();
      ctx.strokeStyle = glow;
      ctx.lineWidth = Math.max(14, CELL_SIZE * 0.36);
      ctx.shadowColor = '#86efac';
      ctx.shadowBlur = Math.max(24, CELL_SIZE * 0.55);
      ctx.shadowOffsetX = 0;
      ctx.shadowOffsetY = 0;
      ctx.globalAlpha = 0.65;
      ctx.stroke();
      ctx.restore();
      ctx.beginPath();
      ctx.moveTo(edgePts[0][0], edgePts[0][1]);
      for (let i = 1; i < edgePts.length; i++) ctx.lineTo(edgePts[i][0], edgePts[i][1]);
      ctx.strokeStyle = core;
      ctx.lineWidth = Math.max(7, CELL_SIZE * 0.2);
      ctx.shadowBlur = 0;
      ctx.globalAlpha = 1;
      ctx.stroke();
    });
    ctx.restore();
  }

  function polylineLengthPx(pathPts) {
    let total = 0;
    for (let i = 1; i < pathPts.length; i++) total += pathDist(pathPts[i - 1], pathPts[i]);
    return total;
  }
  function pointAlongPolylinePx(pathPts, distPx) {
    if (!Array.isArray(pathPts) || pathPts.length < 2) return null;
    let remain = Math.max(0, Number(distPx) || 0);
    for (let i = 1; i < pathPts.length; i++) {
      const p0 = pathPts[i - 1];
      const p1 = pathPts[i];
      const segLen = pathDist(p0, p1);
      if (!(segLen > 1e-6)) continue;
      if (remain <= segLen) {
        const t = remain / segLen;
        return [p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t];
      }
      remain -= segLen;
    }
    return pathPts[pathPts.length - 1];
  }
  function drawPolylineDirectionArrows(pathPts, strokeStyle, arrowFill, lineWidth, spacingPx, headSizePx) {
    if (!Array.isArray(pathPts) || pathPts.length < 2) return;
    const totalLen = polylineLengthPx(pathPts);
    if (!(totalLen > 1e-6)) return;
    const spacing = Math.max(16, spacingPx || 42);
    let arrowCount = 0;
    for (let distPx = spacing * 0.75; distPx < totalLen && arrowCount < PATH_DIRECTION_ARROWS_MAX; distPx += spacing) {
      const tail = pointAlongPolylinePx(pathPts, distPx - Math.max(6, headSizePx * 0.9));
      const tip = pointAlongPolylinePx(pathPts, distPx);
      if (!tail || !tip) continue;
      const dx = tip[0] - tail[0];
      const dy = tip[1] - tail[1];
      const len = Math.hypot(dx, dy);
      if (!(len > 1e-6)) continue;
      const ux = dx / len;
      const uy = dy / len;
      const px = -uy;
      const py = ux;
      const headSize = Math.max(4, headSizePx || 10);
      const baseX = tip[0] - ux * headSize;
      const baseY = tip[1] - uy * headSize;
      ctx.save();
      ctx.fillStyle = arrowFill;
      ctx.strokeStyle = strokeStyle;
      ctx.lineWidth = Math.max(1.5, lineWidth * 0.22);
      ctx.beginPath();
      ctx.moveTo(tip[0], tip[1]);
      ctx.lineTo(baseX + px * headSize * 0.45, baseY + py * headSize * 0.45);
      ctx.lineTo(baseX - px * headSize * 0.45, baseY - py * headSize * 0.45);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
      ctx.restore();
      arrowCount++;
    }
  }
  function drawFlightPathHighlight() {
    const sel = state.selectedObject;
    if (!sel || sel.type !== 'flight' || !sel.obj) return;
    const f = sel.obj;
    if (arrivalAirsideBlocked(f)) return;
    const pathPts = getPathForFlight(f);
    if (!pathPts || pathPts.length < 2) return;
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.translate(state.panX, state.panY);
    ctx.scale(state.scale, state.scale);
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 10;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(pathPts[0][0], pathPts[0][1]);
    for (let i = 1; i < pathPts.length; i++) ctx.lineTo(pathPts[i][0], pathPts[i][1]);
    ctx.stroke();
    drawPolylineDirectionArrows(pathPts, _canvas2dStyle.pathArrivalArrowStroke || 'rgba(252, 165, 165, 0.9)', 'rgba(252, 165, 165, 0.8)', 6, 26.4, 6.6);

    ctx.font = 'bold ' + Math.max(9, CELL_SIZE * 0.35) + 'px system-ui';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'bottom';
    ctx.fillStyle = '#fca5a5';
    function anchorOffPathForLabel(pt, perpPx) {
      if (!pt || !pathPts || pathPts.length < 2) return pt;
      let bestSeg = 0, bestD2 = Infinity;
      for (let si = 0; si < pathPts.length - 1; si++) {
        const near = closestPointOnSegment(pathPts[si], pathPts[si + 1], pt);
        if (!near) continue;
        const d2 = dist2(near, pt);
        if (d2 < bestD2) { bestD2 = d2; bestSeg = si; }
      }
      const p0 = pathPts[bestSeg], p1 = pathPts[bestSeg + 1];
      const dx = p1[0] - p0[0], dy = p1[1] - p0[1];
      const len = Math.hypot(dx, dy) || 1;
      let nx = -dy / len, ny = dx / len;
      if (ny > 0) { nx = -nx; ny = -ny; }
      const d = Math.max(14, perpPx || 22);
      return [pt[0] + nx * d, pt[1] + ny * d];
    }
    function drawSpeedLabel(pt, label) {
      if (!pt) return;
      const ox = 4, oy = -4;
      ctx.fillText(label, pt[0] + ox, pt[1] + oy);
    }
    function drawTouchDownLabel(pt, distM, speedMs) {
      if (!pt) return;
      const a = anchorOffPathForLabel(pt, Math.max(18, CELL_SIZE * 0.55));
      const ox = 2, oy = -6;
      const x = a[0] + ox, yBot = a[1] + oy;
      const lh = Math.max(11, Math.round(CELL_SIZE * 0.36));
      let distPart = '---m';
      if (typeof distM === 'number' && isFinite(distM)) {
        const r = Math.round(distM);
        distPart = (r >= 1000 ? String(r) : String(r).padStart(3, '0')) + 'm';
      }
      let spdPart = '--.-m/s';
      if (typeof speedMs === 'number' && isFinite(speedMs)) {
        spdPart = speedMs.toFixed(1) + 'm/s';
      }
      ctx.textAlign = 'left';
      ctx.textBaseline = 'bottom';
      ctx.strokeStyle = 'rgba(15, 23, 42, 0.92)';
      ctx.lineWidth = 3;
      ctx.lineJoin = 'round';
      const line1 = '(' + distPart + ',  ' + spdPart + ')';
      const line2 = 'Touch Down';
      ctx.strokeText(line1, x, yBot);
      ctx.strokeText(line2, x, yBot - lh);
      ctx.fillStyle = '#fca5a5';
      ctx.fillText(line1, x, yBot);
      ctx.fillText(line2, x, yBot - lh);
    }
    let tdPt = null, retInPt = null, retOutPt = null;
    if (f.arrRunwayIdUsed && typeof getRunwayPointAtDistance === 'function') {
      if (typeof f.arrTdDistM === 'number' && isFinite(f.arrTdDistM)) {
        tdPt = getRunwayPointAtDistance(f.arrRunwayIdUsed, f.arrTdDistM);
      }
      if (typeof f.arrRetDistM === 'number' && isFinite(f.arrRetDistM)) {
        retInPt = getRunwayPointAtDistance(f.arrRunwayIdUsed, f.arrRetDistM);
      }
    }
    if (!retOutPt && f.sampledArrRet) {
      const tw = (state.taxiways || []).find(t => t.id === f.sampledArrRet);
      if (tw && Array.isArray(tw.vertices) && tw.vertices.length) {
        const last = tw.vertices[tw.vertices.length - 1];
        retOutPt = cellToPixel(last.col, last.row);
      }
    }
    if (!tdPt && pathPts.length >= 1) tdPt = pathPts[0];
    if (!retInPt && pathPts.length >= 3) {
      const idxIn = Math.max(1, Math.floor(pathPts.length * 0.4));
      retInPt = pathPts[Math.min(idxIn, pathPts.length - 1)];
    }
    if (!retOutPt && pathPts.length >= 3) {
      const idxOut = Math.max(2, Math.floor(pathPts.length * 0.7));
      retOutPt = pathPts[Math.min(idxOut, pathPts.length - 1)];
    }
    if (tdPt && ((typeof f.arrVTdMs === 'number' && isFinite(f.arrVTdMs)) || (typeof f.arrTdDistM === 'number' && isFinite(f.arrTdDistM)))) {
      drawTouchDownLabel(tdPt, f.arrTdDistM, f.arrVTdMs);
    }
    if (!f.arrRetFailed && typeof f.arrVRetInMs === 'number' && isFinite(f.arrVRetInMs)) {
      drawSpeedLabel(retInPt, 'RET IN ' + f.arrVRetInMs.toFixed(1) + ' m/s');
    }
    if (!f.arrRetFailed && typeof f.arrVRetOutMs === 'number' && isFinite(f.arrVRetOutMs)) {
      drawSpeedLabel(retOutPt, 'RET OUT ' + f.arrVRetOutMs.toFixed(1) + ' m/s');
    }
    ctx.restore();
  }

  function drawDeparturePathHighlight() {
    const sel = state.selectedObject;
    if (!sel || sel.type !== 'flight' || !sel.obj) return;
    const f = sel.obj;
    if (f.noWayDep) return;
    const pathPts = getPathForFlightDeparture(f);
    if (!pathPts || pathPts.length < 2) return;
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.translate(state.panX, state.panY);
    ctx.scale(state.scale, state.scale);
    ctx.strokeStyle = _canvas2dStyle.pathDepartureStroke || '#000000';
    ctx.lineWidth = 4.8;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(pathPts[0][0], pathPts[0][1]);
    for (let i = 1; i < pathPts.length; i++) ctx.lineTo(pathPts[i][0], pathPts[i][1]);
    ctx.stroke();
    drawPolylineDirectionArrows(pathPts, _canvas2dStyle.pathDepartureArrowStroke || '#111827', _canvas2dStyle.pathDepartureArrowStroke || '#111827', 6, 40, 10);
    ctx.restore();
  }

  function drawApproachPreviewPaths2D() {
    if (!state.hasSimulationResult || !state.globalUpdateFresh) return;
    const flights = state.flights || [];
    let f = null;
    for (let i = 0; i < flights.length; i++) {
      const ff = flights[i];
      if (!ff || ff.arrDep === 'Dep' || arrivalAirsideBlocked(ff)) continue;
      const token = ff.token || {};
      const rwId = ff.arrRunwayIdUsed || token.arrRunwayId || token.runwayId || ff.arrRunwayId;
      if (rwId == null || rwId === '') continue;
      f = ff;
      break;
    }
    if (!f) return;
    const token = f.token || {};
    const runwayId = f.arrRunwayIdUsed || token.arrRunwayId || token.runwayId || f.arrRunwayId;
    const rwDir = String(f.arrRunwayDirUsed || 'clockwise');
    const tdDist = touchdownDistMForTimeline(f);
    const anchorDist = arrivalApproachAnchorDistM(runwayId, tdDist);
    const pack = buildStraightApproachPolylineWorld(runwayId, rwDir, anchorDist, APPROACH_OFFSET_WORLD_M);
    let pts;
    if (pack && pack.pts && pack.pts.length >= 2) {
      pts = pack.pts;
    } else {
      const rsPt = getRunwayPointAtDistance(runwayId, anchorDist);
      if (!rsPt) return;
      pts = [approachPointBeforeThresholdJs(runwayId, rwDir, APPROACH_OFFSET_WORLD_M, anchorDist), [rsPt[0], rsPt[1]]];
    }
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.translate(state.panX, state.panY);
    ctx.scale(state.scale, state.scale);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.setLineDash([]);
    ctx.strokeStyle = c2dApproachPreviewStroke();
    ctx.lineWidth = c2dApproachPreviewWidthM();
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (let j = 1; j < pts.length; j++) ctx.lineTo(pts[j][0], pts[j][1]);
    ctx.stroke();
    ctx.restore();
  }

  function drawFlights2D() {
    if (!state.hasSimulationResult || !state.flights.length) return;
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.translate(state.panX, state.panY);
    ctx.scale(state.scale, state.scale);
    const tSecDraw = state.simTimeSec;
    if (typeof prepareLazyTimelinesForCurrentSim === 'function') prepareLazyTimelinesForCurrentSim(tSecDraw);
    state.flights.forEach(f => {
      if (flightBlockedLikeNoWay(f)) return;
      if (!state.globalUpdateFresh) return;
      const pose = getFlightPoseAtTimeForDraw(f, tSecDraw);
      if (!pose) return;
      const x = pose.x, y = pose.y, dx = pose.dx, dy = pose.dy;
      const len = Math.hypot(dx, dy) || 1;
      const nx = dx / len, ny = dy / len;
      const silN = Number(_acSil.noseX), silWR = Number(_acSil.wingRearX), silUY = Number(_acSil.wingUpperY);
      const silTN = Number(_acSil.tailNeckX), silLY = Number(_acSil.wingLowerY);
      const nX = isFinite(silN) ? silN : 0.6;
      const wRx = isFinite(silWR) ? silWR : -0.5;
      const uY = isFinite(silUY) ? silUY : 0.35;
      const tX = isFinite(silTN) ? silTN : -0.3;
      const lY = isFinite(silLY) ? silLY : -0.35;
      const useDetailSil = _ac2d.useDetailedSilhouette === true;
      
      const silhouette2D = [
        [0.86, 0],
        [0.74, 0.038], [0.55, 0.046], [0.35, 0.048], [0.16, 0.05],
        [-0.16, 0.5],
        [-0.22, 0.5],
        [-0.38, 0.09], [-0.52, 0.056], [-0.66, 0.046],
        [-0.76, 0.15],
        [-0.82, 0.036], [-0.88, 0],
        [-0.82, -0.036],
        [-0.76, -0.15],
        [-0.66, -0.046], [-0.52, -0.056], [-0.38, -0.09],
        [-0.22, -0.5],
        [-0.16, -0.5],
        [0.16, -0.05], [0.35, -0.048], [0.55, -0.046], [0.74, -0.038],
      ];
      let scaleX, scaleY, sizeRef;
      if (useDetailSil) {
        let minXn = Infinity, maxXn = -Infinity, maxYy = 0;
        for (let si = 0; si < silhouette2D.length; si++) {
          const px = silhouette2D[si][0], py = silhouette2D[si][1];
          minXn = Math.min(minXn, px);
          maxXn = Math.max(maxXn, px);
          maxYy = Math.max(maxYy, Math.abs(py));
        }
        const lenNorm = Math.max(1e-9, maxXn - minXn);
        const wingNorm = Math.max(1e-9, 2 * maxYy);
        scaleX = AIRCRAFT_FUSELAGE_LENGTH_M / lenNorm;
        scaleY = AIRCRAFT_WINGSPAN_M / wingNorm;
        sizeRef = 0.5 * Math.hypot(AIRCRAFT_FUSELAGE_LENGTH_M, AIRCRAFT_WINGSPAN_M);
      } else {
        const xs = [nX, wRx, tX];
        const minXn = Math.min(xs[0], xs[1], xs[2]);
        const maxXn = Math.max(xs[0], xs[1], xs[2]);
        const lenNorm = Math.max(1e-9, maxXn - minXn);
        const wingNorm = Math.max(1e-9, uY + lY);
        scaleX = AIRCRAFT_FUSELAGE_LENGTH_M / lenNorm;
        scaleY = AIRCRAFT_WINGSPAN_M / wingNorm;
        sizeRef = 0.5 * Math.hypot(AIRCRAFT_FUSELAGE_LENGTH_M, AIRCRAFT_WINGSPAN_M);
      }
      const outW = Number(_ac2d.outlineWidth);
      const outlineWidth = (isFinite(outW) && outW > 0) ? outW : 0;
      const outlineColor = _ac2d.outlineColor || '';
      const isFlightSel = state.selectedObject && state.selectedObject.type === 'flight' && state.selectedObject.id === f.id;
      if (FLIGHT_TRAIL_LENGTH_M > 0 && !isFlightTrailHiddenAtSimTime(f, tSecDraw)) {
        const trailPts = getFlightTrailPolylineBackward(f, tSecDraw, FLIGHT_TRAIL_LENGTH_M);
        if (trailPts.length >= 2) {
          ctx.save();
          const x0 = trailPts[0][0], y0 = trailPts[0][1];
          const x1 = trailPts[trailPts.length - 1][0], y1 = trailPts[trailPts.length - 1][1];
          const g = ctx.createLinearGradient(x0, y0, x1, y1);
          const cFar = c2dSimFlightTrailStrokeEnd();
          const cNearAc = c2dSimFlightTrailStroke();
          g.addColorStop(0, cFar);
          g.addColorStop(0.42, cNearAc);
          g.addColorStop(1, cNearAc);
          ctx.strokeStyle = g;
          ctx.lineWidth = c2dSimFlightTrailLineWidth();
          ctx.lineCap = 'round';
          ctx.lineJoin = 'round';
          ctx.setLineDash([]);
          ctx.beginPath();
          ctx.moveTo(trailPts[0][0], trailPts[0][1]);
          for (let ti = 1; ti < trailPts.length; ti++) ctx.lineTo(trailPts[ti][0], trailPts[ti][1]);
          ctx.stroke();
          ctx.restore();
        }
      }
      if (isFlightPreTouchdownForDraw(f, tSecDraw)) {
        const rH = Math.max(sizeRef * 0.58, 8);
        ctx.save();
        ctx.beginPath();
        ctx.arc(x, y, rH, 0, Math.PI * 2);
        ctx.fillStyle = c2dSimPreTouchdownHaloFill();
        ctx.fill();
        ctx.strokeStyle = c2dSimPreTouchdownHaloStroke();
        ctx.lineWidth = 2;
        ctx.shadowColor = c2dSimPreTouchdownHaloStroke();
        ctx.shadowBlur = c2dSimPreTouchdownHaloBlur();
        ctx.stroke();
        ctx.restore();
      }
      if (isFlightSel) {
        ctx.save();
        ctx.beginPath();
        ctx.arc(x, y, sizeRef * 0.62, 0, Math.PI * 2);
        ctx.strokeStyle = c2dObjectSelectedStroke();
        ctx.lineWidth = 2.5;
