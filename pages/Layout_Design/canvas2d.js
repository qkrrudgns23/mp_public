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
      const rwOpDir = normalizeRwDirectionValue(getTaxiwayDirection(rw));

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
                const standPt = getStandApronTaxiwayAttachWorldPx(pbb);
                const mids = (Array.isArray(lk.midVertices) ? lk.midVertices : []).map(function(v) { return cellToPixel(Number(v.col), Number(v.row)); });
                const chain = [standPt].concat(mids).concat([p]);
                apronNodeStand.push({ nodeP: p, standPt, standId: lk.pbbId, chain, linkId: lk.id || 'apron_link' });
              }
            }
          });
        }
        {
          const ptHp = obj.pathType;
          if (ptHp === 'runway_exit' || ptHp === 'taxiway' || ptHp === 'apron_taxiway' || ptHp === 'general_queue_taxiway') {
            const csH = (typeof CELL_SIZE === 'number' && isFinite(CELL_SIZE) && CELL_SIZE > 0) ? CELL_SIZE : 20;
            const hpTolD2 = Math.max(SPLIT_TOL_D2, (csH * 0.35) * (csH * 0.35));
            (state.holdingPoints || []).forEach(function(hp) {
              if (!hp) return;
              const k = (typeof normalizeHoldingPointKind === 'function') ? normalizeHoldingPointKind(hp.hpKind) : String(hp.hpKind || '').trim();
              if (ptHp === 'runway_exit') {
                if (k !== 'runway_holding') return;
              } else {
                if (k !== 'intermediate') return;
              }
              if (typeof hp.x !== 'number' || typeof hp.y !== 'number' || !isFinite(hp.x) || !isFinite(hp.y)) return;
              const pr = projectOnSegment(a, b, [hp.x, hp.y]);
              if (pr.t >= 0 && pr.t <= 1 && dist2(pr.p, [hp.x, hp.y]) <= hpTolD2) {
                junctions.push({ tAlong: seg + pr.t, p: pr.p });
              }
            });
          }
        }
        {
          const ptTs = obj.pathType;
          if (
            ptTs === 'runway_exit' ||
            ptTs === 'taxiway' ||
            ptTs === 'apron_taxiway' ||
            ptTs === 'runway_taxiway' ||
            ptTs === 'general_queue_taxiway'
          ) {
            (state.tempStands || []).forEach(function(st) {
              if (!st) return;
              const corners = getRemoteStandCorners(st);
              if (!corners || corners.length < 4) return;
              for (let ei = 0; ei < 4; ei++) {
                const c = corners[ei], d = corners[(ei + 1) % 4];
                const isec = segmentSegmentIntersection(a, b, c, d);
                if (isec) {
                  const pr = projectOnSegment(a, b, isec.p);
                  if (pr.t >= 0 && pr.t <= 1) junctions.push({ tAlong: seg + pr.t, p: pr.p });
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
                    [c, d].forEach(function(q) {
                      if (dist2(a, q) <= SPLIT_TOL_D2 || dist2(b, q) <= SPLIT_TOL_D2) {
                        const prq = projectOnSegment(a, b, q);
                        if (prq.t >= 0 && prq.t <= 1) junctions.push({ tAlong: seg + prq.t, p: prq.p });
                      }
                    });
                  }
                }
              }
            });
          }
        }
      }
      if (obj.pathType === 'general_queue_taxiway') {
        queueTaxiwayAutoJunctionMarkersAlong(obj, QUEUE_TAXIWAY_JUNCTION_SPACING_M).forEach(function(qj) {
          junctions.push(qj);
        });
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
      const isTaxiway = obj.pathType === 'taxiway' || obj.pathType === 'apron_taxiway' || obj.pathType === 'general_queue_taxiway';
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

  /** RET gate filtering only: exit direction is not chosen via taxi Dijkstra. Use flight token / defaults. */
  function probePreferredArrivalRunwayDir(f) {
    void f;
    return 'both';
  }
  function resolveArrivalRunwayDirForRetGate(f) {
    const fromFlight = normalizeRwDirectionValue(f.arrRunwayDirUsed);
    if (fromFlight === 'clockwise' || fromFlight === 'counter_clockwise') return fromFlight;
    const probed = probePreferredArrivalRunwayDir(f);
    if (probed === 'clockwise' || probed === 'counter_clockwise') return probed;
    return 'both';
  }

  function graphPathArrival(f) {
    if (f) {
      f.noWayArr = false;
      delete f._noWayArrDetail;
    }
    return null;
  }

  function graphPathDeparture(f, opts) {
    if (f) {
      f.noWayDep = false;
      delete f._noWayDepDetail;
    }
    return null;
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
    if (!f) return null;
    resolveStand(f);
    delete f.cachedArrPathPts;
    delete f._pathPolylineArrRetKey;
    return null;
  }

  function getPathForFlightDeparture(f) {
    if (!f) return null;
    resolveStand(f);
    delete f.cachedDepPathPts;
    return null;
  }

  function ensureFlightPaths(f) {
    void f;
  }

  function findStandById(id) {
    return (state.pbbStands || []).find(function(s) { return s.id === id; }) ||
           (state.remoteStands || []).find(function(s) { return s.id === id; }) ||
           (state.tempStands || []).find(function(s) { return s.id === id; });
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
    void flight;
    void direction;
    return { pts: null, timeline: null };
  }

  const FLIGHT_PATH_PROGRESS_PCT_START = 22;
  const FLIGHT_PATH_PROGRESS_PCT_END = 48;
  const PATH_DIRECTION_ARROWS_MAX = 48;
  function updateAllFlightPaths(onDone) {
    draw();
    if (typeof onDone === 'function') onDone();
  }

  function drawPathJunctions() {
    if (!state.layers.junction) return;
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
    const rGreen = r * 0.7;
    ctx.fillStyle = '#ef4444';
    redJunctions.forEach(p => {
      ctx.beginPath();
      ctx.arc(p[0], p[1], r, 0, Math.PI * 2);
      ctx.fill();
    });
    ctx.fillStyle = '#22c55e';
    connectedJunctions.forEach(p => {
      ctx.beginPath();
      ctx.arc(p[0], p[1], rGreen, 0, Math.PI * 2);
      ctx.fill();
    });
    // Edge distance numeric labels are intentionally hidden in layout view.
    ctx.restore();
  }

  function drawQueueTaxiwayLaneMarkers() {
    if (!state.layers.junction) return;
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.translate(state.panX, state.panY);
    ctx.scale(state.scale, state.scale);
    const r = Math.max(3.5, CELL_SIZE * 0.22) * LAYOUT_VERTEX_DOT_SCALE;
    (state.taxiways || []).forEach(function(tw) {
      if (!tw || tw.pathType !== 'general_queue_taxiway' || !tw.vertices || tw.vertices.length < 2) return;
      const jm = queueTaxiwayAutoJunctionMarkersAlong(tw, QUEUE_TAXIWAY_JUNCTION_SPACING_M);
      for (let j = 0; j < jm.length; j++) {
        const xy = jm[j].p;
        ctx.beginPath();
        ctx.arc(xy[0], xy[1], r, 0, Math.PI * 2);
        ctx.fillStyle = '#22c55e';
        ctx.fill();
        ctx.strokeStyle = '#ef4444';
        ctx.lineWidth = 1.8;
        ctx.stroke();
      }
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

  const PRO_SIM_PHASE_Z = {
    Landing: 0,
    Arr_taxi: 1,
    Arr_taxi_occupied: 1,
    Dep_taxi: 2,
    Holding_lineup: 3,
    Lineup_departure: 4,
  };
  function proSimPhaseStrokeStyle(phaseRaw) {
    const p = (phaseRaw != null && String(phaseRaw).trim()) ? String(phaseRaw).trim() : 'Landing';
    if (p === 'Arr_taxi_occupied') {
      return { wMul: 1.72, stroke: '#a855f7' };
    }
    if (p === 'Arr_taxi') {
      return { wMul: 1.72, stroke: '#3b82f6' };
    }
    if (p === 'Dep_taxi' || p === 'Holding_lineup') {
      return { wMul: 0.58, stroke: '#ef4444' };
    }
    if (p === 'Lineup_departure') {
      return { wMul: 0.45, stroke: '#f97316' };
    }
    return { wMul: 1.72, stroke: '#22c55e' };
  }
  function drawProSimSegmentArrows(edgePts, arrowFill, spacingPx, headSizePx) {
    if (!Array.isArray(edgePts) || edgePts.length < 2) return;
    const spacing = Math.max(14, spacingPx || 36);
    let count = 0;
    const headSize = Math.max(4, headSizePx || 10);
    let refUx = 0;
    let refUy = 0;
    let refSet = false;
    for (let i = 1; i < edgePts.length && !refSet; i++) {
      const p0 = edgePts[i - 1];
      const p1 = edgePts[i];
      const segLen = pathDist(p0, p1);
      if (segLen < 1e-6) continue;
      refUx = (p1[0] - p0[0]) / segLen;
      refUy = (p1[1] - p0[1]) / segLen;
      refSet = true;
    }
    if (!refSet) return;
    for (let i = 1; i < edgePts.length && count < PATH_DIRECTION_ARROWS_MAX; i++) {
      const p0 = edgePts[i - 1];
      const p1 = edgePts[i];
      const segLen = pathDist(p0, p1);
      if (segLen < 1e-6) continue;
      const ux = (p1[0] - p0[0]) / segLen;
      const uy = (p1[1] - p0[1]) / segLen;
      if (ux * refUx + uy * refUy < -0.08) continue;
      const px = -uy;
      const py = ux;
      for (let d = spacing * 0.55; d < segLen - headSize * 0.35 && count < PATH_DIRECTION_ARROWS_MAX; d += spacing) {
        const tTip = d / segLen;
        const tipx = p0[0] + (p1[0] - p0[0]) * tTip;
        const tipy = p0[1] + (p1[1] - p0[1]) * tTip;
        const baseX = tipx - ux * headSize;
        const baseY = tipy - uy * headSize;
        ctx.save();
        ctx.fillStyle = arrowFill;
        ctx.beginPath();
        ctx.moveTo(tipx, tipy);
        ctx.lineTo(baseX + px * headSize * 0.45, baseY + py * headSize * 0.45);
        ctx.lineTo(baseX - px * headSize * 0.45, baseY - py * headSize * 0.45);
        ctx.closePath();
        ctx.fill();
        ctx.restore();
        count++;
      }
    }
  }
  function orientProSimEdgePts(edgePts, prevEnd, prevUx, prevUy) {
    let pts = edgePts.slice();
    if (pts.length < 2) return pts;
    if (prevEnd) {
      const d0 = dist2(pts[0], prevEnd);
      const d1 = dist2(pts[pts.length - 1], prevEnd);
      if (d1 + 9 < d0) {
        pts.reverse();
      } else if (Math.abs(d0 - d1) <= 36 && prevUx != null && prevUy != null) {
        let vx = 0;
        let vy = 0;
        for (let i = 1; i < pts.length; i++) {
          const dx = pts[i][0] - pts[i - 1][0];
          const dy = pts[i][1] - pts[i - 1][1];
          const sl = Math.hypot(dx, dy);
          if (sl > 1e-6) {
            vx = dx / sl;
            vy = dy / sl;
            break;
          }
        }
        if (vx * prevUx + vy * prevUy < -0.15) pts.reverse();
      }
    } else if (prevUx != null && prevUy != null) {
      let vx = 0;
      let vy = 0;
      for (let i = 1; i < pts.length; i++) {
        const dx = pts[i][0] - pts[i - 1][0];
        const dy = pts[i][1] - pts[i - 1][1];
        const sl = Math.hypot(dx, dy);
        if (sl > 1e-6) {
          vx = dx / sl;
          vy = dy / sl;
          break;
        }
      }
      if (vx * prevUx + vy * prevUy < -0.15) pts.reverse();
    }
    return pts;
  }
  function proSimOutgoingUnit(edgePts) {
    if (!edgePts || edgePts.length < 2) return { ux: null, uy: null };
    for (let i = edgePts.length - 1; i >= 1; i--) {
      const dx = edgePts[i][0] - edgePts[i - 1][0];
      const dy = edgePts[i][1] - edgePts[i - 1][1];
      const sl = Math.hypot(dx, dy);
      if (sl > 1e-6) return { ux: dx / sl, uy: dy / sl };
    }
    return { ux: null, uy: null };
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
    const baseW = Math.max(4.2, CELL_SIZE * 0.148);
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.translate(state.panX, state.panY);
    ctx.scale(state.scale, state.scale);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.shadowBlur = 0;
    ctx.shadowColor = 'transparent';
    let prevEnd = null;
    let prevUx = null;
    let prevUy = null;
    let lastDrawnKey = null;
    let seqIx = 0;
    const drawList = [];
    ids.forEach(function(entry) {
      let key = '';
      let phase = 'Landing';
      if (entry != null) {
        if (typeof entry === 'string' || typeof entry === 'number') {
          key = String(entry).trim();
        } else if (typeof entry === 'object') {
          const rawId = entry.edge_id != null ? entry.edge_id : entry.id;
          key = rawId != null ? String(rawId).trim() : '';
          if (entry.phase != null) phase = String(entry.phase).trim() || 'Landing';
        }
      }
      if (key && key === lastDrawnKey) {
        return;
      }
      const st = proSimPhaseStrokeStyle(phase);
      const lineW = baseW * st.wMul;
      const e = key ? byId[key] : null;
      if (!e) return;
      let rawPts = (e.pts && e.pts.length >= 2) ? e.pts.slice() : [[e.x1, e.y1], [e.x2, e.y2]];
      let edgePts = orientProSimEdgePts(rawPts, prevEnd, prevUx, prevUy);
      const z = Object.prototype.hasOwnProperty.call(PRO_SIM_PHASE_Z, phase) ? PRO_SIM_PHASE_Z[phase] : 0;
      drawList.push({
        edgePts: edgePts,
        st: st,
        lineW: lineW,
        z: z,
        seq: seqIx++,
      });
      prevEnd = edgePts[edgePts.length - 1];
      const ou = proSimOutgoingUnit(edgePts);
      if (ou.ux != null) {
        prevUx = ou.ux;
        prevUy = ou.uy;
      }
      lastDrawnKey = key;
    });
    drawList.sort(function(a, b) {
      if (a.z !== b.z) return a.z - b.z;
      return a.seq - b.seq;
    });
    drawList.forEach(function(item) {
      const edgePts = item.edgePts;
      ctx.beginPath();
      ctx.moveTo(edgePts[0][0], edgePts[0][1]);
      for (let i = 1; i < edgePts.length; i++) ctx.lineTo(edgePts[i][0], edgePts[i][1]);
      ctx.strokeStyle = item.st.stroke;
      ctx.lineWidth = item.lineW;
      ctx.globalAlpha = 0.92;
      ctx.stroke();
      ctx.globalAlpha = 1;
      drawProSimSegmentArrows(
        edgePts,
        'rgba(250, 250, 250, 0.82)',
        Math.max(20, CELL_SIZE * 0.34),
        Math.max(4.5, CELL_SIZE * 0.135)
      );
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
  function drawPolylineDirectionArrows(pathPts, strokeStyle, arrowFill, lineWidth, spacingPx, headSizePx, omitStroke) {
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
      if (!omitStroke) ctx.stroke();
      ctx.restore();
      arrowCount++;
    }
  }
  function drawFlightPathHighlight() {
    return;
  }

  function drawDeparturePathHighlight() {
    return;
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
    state.flights.forEach(f => {
      if (flightBlockedLikeNoWay(f)) return;
      const pose = getFlightPoseAtTimeForDraw(f, tSecDraw);
      if (!pose) return;
      const x = pose.x, y = pose.y, dx = pose.dx, dy = pose.dy;
      const len = Math.hypot(dx, dy) || 1;
