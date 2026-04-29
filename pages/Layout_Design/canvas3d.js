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
  function taxiTimeAtCumulativeDistanceM(pts, tStart, tEnd, sTarget, velForSeg) {
    if (!pts || pts.length < 2 || !(sTarget > 1e-6) || tEnd <= tStart + 1e-9) return tStart;
    const lengths = [];
    for (let i = 0; i < pts.length - 1; i++) lengths.push(pathDist(pts[i], pts[i + 1]));
    const rawDts = [];
    for (let i = 0; i < lengths.length; i++) {
      const v = Math.max(1, velForSeg(i, pts[i], pts[i + 1]));
      rawDts.push(lengths[i] < 1e-9 ? 0 : lengths[i] / v);
    }
    const rawTotal = rawDts.reduce(function(s, x) { return s + x; }, 0);
    if (rawTotal < 1e-9) return tStart;
    const totalLen = lengths.reduce(function(s, l) { return s + l; }, 0);
    const sClamped = Math.min(sTarget, totalLen);
    if (sClamped >= totalLen - 1e-6) return tEnd;
    const scale = (tEnd - tStart) / rawTotal;
    let distAcc = 0;
    let rawAcc = 0;
    for (let i = 0; i < lengths.length; i++) {
      if (distAcc + lengths[i] >= sClamped) {
        const u = (sClamped - distAcc) / (lengths[i] || 1);
        return tStart + (rawAcc + u * rawDts[i]) * scale;
      }
      distAcc += lengths[i];
      rawAcc += rawDts[i];
    }
    return tEnd;
  }
  function depNoseExitsApronCumulativeM(f, pathPts) {
    if (!pathPts || pathPts.length < 2) return 0;
    const standId = f && (f.standId != null && f.standId !== '' ? f.standId : (f.token && f.token.apronId));
    if (standId == null || standId === '') return 0;
    const links = state.apronLinks || [];
    const lk = links.find(function(l) { return l && String(l.pbbId) === String(standId); });
    if (!lk || lk.tx == null || lk.ty == null) return 0;
    const jx = Number(lk.tx), jy = Number(lk.ty);
    if (!isFinite(jx) || !isFinite(jy)) return 0;
    const cum = cumulativeDistAlongPolylineToPoint(pathPts, [jx, jy]);
    if (!cum) return 0;
    if (cum.d2 > 35 * 35) return 0;
    const L = polylineTotalLength(pathPts);
    return Math.min(cum.distAlong, Math.max(0, L - 1e-3));
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
    const mergeR = PATH_JUNCTION_MERGE_RADIUS_PX * 3.5;
    const mergeR2 = mergeR * mergeR;
    const conn = gCache.connectedJunctions || gCache.junctions || [];
    const disc = gCache.disconnectedValidJunctions;
    let i;
    for (i = 0; i < conn.length; i++) {
      if (dist2(p, conn[i]) <= mergeR2) return '#22c55e';
    }
    if (disc && disc.length) {
      for (i = 0; i < disc.length; i++) {
        if (dist2(p, disc[i]) <= mergeR2) return '#ef4444';
      }
    }
    return '#22c55e';
  }
  function segmentWorldAabbPadded(a, b, pad) {
    const p = Number(pad) && isFinite(pad) ? pad : 0;
    return {
      minX: Math.min(a[0], b[0]) - p,
      maxX: Math.max(a[0], b[0]) + p,
      minY: Math.min(a[1], b[1]) - p,
      maxY: Math.max(a[1], b[1]) + p
    };
  }
  function aabbWorldIntersects2D(ax, bx) {
    if (!ax || !bx) return true;
    return !(ax.maxX < bx.minX || ax.minX > bx.maxX || ax.maxY < bx.minY || ax.minY > bx.maxY);
  }
  function collectPathJunctionWorldPointsForTaxiway(obj, pathList) {
    if (!obj || !pathList || !pathList.length) return [];
    const pts = getOrderedPoints(obj);
    if (!pts || pts.length < 2) return [];
    const junctions = [];
    const segPad = CELL_SIZE * 32;
    for (let seg = 0; seg < pts.length - 1; seg++) {
      const a = pts[seg], b = pts[seg + 1];
      const segBox = segmentWorldAabbPadded(a, b, segPad);
      pathList.forEach(function(other) {
        if (!other || other.id === obj.id) return;
        const otherBox = taxiwayWorldAabb(other);
        if (otherBox && !aabbWorldIntersects2D(segBox, otherBox)) return;
        const otherOrd = getOrderedPoints(other);
        if (!otherOrd || otherOrd.length < 2) return;
        for (let oseg = 0; oseg < otherOrd.length - 1; oseg++) {
          const c = otherOrd[oseg], d = otherOrd[oseg + 1];
          const isec = segmentSegmentIntersection(a, b, c, d);
          if (isec) {
            const pr = projectOnSegment(a, b, isec.p);
            junctions.push({ tAlong: seg + pr.t, p: pr.p });
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
        otherOrd.forEach(function(q) {
          if (!pointOnSegmentStrict(a, b, q)) return;
          const prq = projectOnSegment(a, b, q);
          junctions.push({ tAlong: seg + prq.t, p: prq.p });
        });
      });
      const isRunway = obj.pathType === 'runway';
      if (!isRunway) {
        (state.apronLinks || []).forEach(function(lk) {
          if (lk.taxiwayId !== obj.id || lk.tx == null || lk.ty == null) return;
          const linkPt = [Number(lk.tx), Number(lk.ty)];
          const pr = projectOnSegment(a, b, linkPt);
          if (pr.t >= 0 && pr.t <= 1 && dist2(pr.p, linkPt) <= SPLIT_TOL_D2) {
            junctions.push({ tAlong: seg + pr.t, p: pr.p });
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
            const prh = projectOnSegment(a, b, [hp.x, hp.y]);
            if (prh.t >= 0 && prh.t <= 1 && dist2(prh.p, [hp.x, hp.y]) <= hpTolD2) {
              junctions.push({ tAlong: seg + prh.t, p: prh.p });
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
              const isec2 = segmentSegmentIntersection(a, b, c, d);
              if (isec2) {
                const pr2 = projectOnSegment(a, b, isec2.p);
                if (pr2.t >= 0 && pr2.t <= 1) junctions.push({ tAlong: seg + pr2.t, p: pr2.p });
              } else {
                const ov2 = collinearSegmentOverlapOnAB(a, b, c, d);
                if (ov2) {
                  const ax2 = a[0], ay2 = a[1], bx2 = b[0], by2 = b[1];
                  const dx2 = bx2 - ax2, dy2 = by2 - ay2;
                  const p0b = [ax2 + ov2.t0 * dx2, ay2 + ov2.t0 * dy2];
                  const p1b = [ax2 + ov2.t1 * dx2, ay2 + ov2.t1 * dy2];
                  const pr0b = projectOnSegment(a, b, p0b);
                  junctions.push({ tAlong: seg + pr0b.t, p: pr0b.p });
                  if (dist2(p0b, p1b) > SPLIT_TOL_D2) {
                    const pr1b = projectOnSegment(a, b, p1b);
                    junctions.push({ tAlong: seg + pr1b.t, p: pr1b.p });
                  }
                } else {
                  [c, d].forEach(function(q) {
                    if (dist2(a, q) <= SPLIT_TOL_D2 || dist2(b, q) <= SPLIT_TOL_D2) {
                      const prq2 = projectOnSegment(a, b, q);
                      if (prq2.t >= 0 && prq2.t <= 1) junctions.push({ tAlong: seg + prq2.t, p: prq2.p });
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
        junctions.push({ tAlong: qj.tAlong, p: qj.p });
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
    const raw = [];
    for (let ji = 0; ji < junctions.length; ji++) {
      const j = junctions[ji];
      const p = j && j.p;
      if (p && isFinite(p[0]) && isFinite(p[1])) raw.push(p);
    }
    return mergeNearbyPathPointsForDraw(raw, PATH_JUNCTION_MERGE_RADIUS_PX);
  }

  /** World-space polyline for apron link in progress (matches drawApronTaxiwayLinks draft). */
  function getApronLinkDrawingDraftWorldPts() {
    if (!state.apronLinkDrawing || !state.apronLinkTemp) return null;
    const t = state.apronLinkTemp;
    const ptsPx = [];
    if (t.kind === 'pbb' || t.kind === 'remote') {
      const st = findStandById(t.standId);
      if (st) ptsPx.push(getStandApronTaxiwayAttachWorldPx(st));
    } else if (t.kind === 'taxiway') {
      if (isFinite(Number(t.x)) && isFinite(Number(t.y))) ptsPx.push([Number(t.x), Number(t.y)]);
    }
    (state.apronLinkMidpoints || []).forEach(function(c) {
      if (c && isFinite(Number(c.x)) && isFinite(Number(c.y))) ptsPx.push([Number(c.x), Number(c.y)]);
      else ptsPx.push(cellToPixel(Number(c.col), Number(c.row)));
    });
    const hoverApron = (state.apronLinkPointerWorld && state.apronLinkPointerWorld.length >= 2 &&
      isFinite(state.apronLinkPointerWorld[0]) && isFinite(state.apronLinkPointerWorld[1]))
      ? state.apronLinkPointerWorld
      : null;
    if (ptsPx.length < 1) return null;
    if (ptsPx.length < 2 && !hoverApron) return null;
    const full = hoverApron ? ptsPx.concat([hoverApron]) : ptsPx.slice();
    return dedupePathPoints(full);
  }

  /** Junction world points where an open polyline crosses taxiway centerlines (same geometry as in-progress taxiway overlay). */
  function collectPathJunctionWorldPointsForOpenPolyline(polyPts, pathList) {
    if (!polyPts || polyPts.length < 2 || !pathList || !pathList.length) return [];
    const junctions = [];
    const segPad = CELL_SIZE * 32;
    for (let seg = 0; seg < polyPts.length - 1; seg++) {
      const a = polyPts[seg], b = polyPts[seg + 1];
      const segBox = segmentWorldAabbPadded(a, b, segPad);
      pathList.forEach(function(other) {
        if (!other) return;
        const otherBox = taxiwayWorldAabb(other);
        if (otherBox && !aabbWorldIntersects2D(segBox, otherBox)) return;
        const otherOrd = getOrderedPoints(other);
        if (!otherOrd || otherOrd.length < 2) return;
        for (let oseg = 0; oseg < otherOrd.length - 1; oseg++) {
          const c = otherOrd[oseg], d = otherOrd[oseg + 1];
          const isec = segmentSegmentIntersection(a, b, c, d);
          if (isec) {
            const pr = projectOnSegment(a, b, isec.p);
            junctions.push({ tAlong: seg + pr.t, p: pr.p });
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
        otherOrd.forEach(function(q) {
          if (!pointOnSegmentStrict(a, b, q)) return;
          const prq = projectOnSegment(a, b, q);
          junctions.push({ tAlong: seg + prq.t, p: prq.p });
        });
      });
      {
        const csH = (typeof CELL_SIZE === 'number' && isFinite(CELL_SIZE) && CELL_SIZE > 0) ? CELL_SIZE : 20;
        const hpTolD2 = Math.max(SPLIT_TOL_D2, (csH * 0.35) * (csH * 0.35));
        (state.holdingPoints || []).forEach(function(hp) {
          if (!hp) return;
          const k = (typeof normalizeHoldingPointKind === 'function') ? normalizeHoldingPointKind(hp.hpKind) : String(hp.hpKind || '').trim();
          if (k !== 'intermediate') return;
          if (typeof hp.x !== 'number' || typeof hp.y !== 'number' || !isFinite(hp.x) || !isFinite(hp.y)) return;
          const prh = projectOnSegment(a, b, [hp.x, hp.y]);
          if (prh.t >= 0 && prh.t <= 1 && dist2(prh.p, [hp.x, hp.y]) <= hpTolD2) {
            junctions.push({ tAlong: seg + prh.t, p: prh.p });
          }
        });
      }
      (state.tempStands || []).forEach(function(st) {
        if (!st) return;
        const corners = getRemoteStandCorners(st);
        if (!corners || corners.length < 4) return;
        for (let ei = 0; ei < 4; ei++) {
          const c = corners[ei], d = corners[(ei + 1) % 4];
          const isec2 = segmentSegmentIntersection(a, b, c, d);
          if (isec2) {
            const pr2 = projectOnSegment(a, b, isec2.p);
            if (pr2.t >= 0 && pr2.t <= 1) junctions.push({ tAlong: seg + pr2.t, p: pr2.p });
          } else {
            const ov2 = collinearSegmentOverlapOnAB(a, b, c, d);
            if (ov2) {
              const ax2 = a[0], ay2 = a[1], bx2 = b[0], by2 = b[1];
              const dx2 = bx2 - ax2, dy2 = by2 - ay2;
              const p0b = [ax2 + ov2.t0 * dx2, ay2 + ov2.t0 * dy2];
              const p1b = [ax2 + ov2.t1 * dx2, ay2 + ov2.t1 * dy2];
              const pr0b = projectOnSegment(a, b, p0b);
              junctions.push({ tAlong: seg + pr0b.t, p: pr0b.p });
              if (dist2(p0b, p1b) > SPLIT_TOL_D2) {
                const pr1b = projectOnSegment(a, b, p1b);
                junctions.push({ tAlong: seg + pr1b.t, p: pr1b.p });
              }
            } else {
              [c, d].forEach(function(q) {
                if (dist2(a, q) <= SPLIT_TOL_D2 || dist2(b, q) <= SPLIT_TOL_D2) {
                  const prq2 = projectOnSegment(a, b, q);
                  if (prq2.t >= 0 && prq2.t <= 1) junctions.push({ tAlong: seg + prq2.t, p: prq2.p });
                }
              });
            }
          }
        }
      });
    }
    const raw = [];
    for (let ji = 0; ji < junctions.length; ji++) {
      const j = junctions[ji];
      const p = j && j.p;
      if (p && isFinite(p[0]) && isFinite(p[1])) raw.push(p);
    }
    return mergeNearbyPathPointsForDraw(raw, PATH_JUNCTION_MERGE_RADIUS_PX);
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
    function segmentBBoxInflated2d(a, b, pad) {
      const p = pad || 0;
      const ax = a[0], ay = a[1], bx = b[0], by = b[1];
      return {
        minX: Math.min(ax, bx) - p,
        maxX: Math.max(ax, bx) + p,
        minY: Math.min(ay, by) - p,
        maxY: Math.max(ay, by) + p
      };
    }
    function bboxesOverlap2d(A, B) {
      if (!A || !B) return true;
      return !(A.maxX < B.minX || B.maxX < A.minX || A.maxY < B.minY || B.maxY < A.minY);
    }
    const orderedPtsCache = new Array(pathList.length);
    for (let _ci = 0; _ci < pathList.length; _ci++) {
      orderedPtsCache[_ci] = getOrderedPoints(pathList[_ci]);
    }
    const apronNodeStand = [];
    const minD2 = 1e-6;
    pathList.forEach(function(obj, objIdx) {
      if (omitOtherRunwayExits && selectedArrRetId != null && obj && obj.pathType === 'runway_exit' && obj.id !== selectedArrRetId) return;
      const pts = orderedPtsCache[objIdx];
      if (!pts || pts.length < 2) return;
      const junctions = [];
      for (let seg = 0; seg < pts.length - 1; seg++) {
        const a = pts[seg], b = pts[seg+1];
        const segAbBbox = segmentBBoxInflated2d(a, b, 1e-6);
        for (let oi = 0; oi < pathList.length; oi++) {
          const other = pathList[oi];
          if (other.id === obj.id) continue;
          const otherOrd = orderedPtsCache[oi];
          if (!otherOrd || otherOrd.length < 2) continue;
          for (let oseg = 0; oseg < otherOrd.length - 1; oseg++) {
            const c = otherOrd[oseg], d = otherOrd[oseg+1];
            if (!bboxesOverlap2d(segAbBbox, segmentBBoxInflated2d(c, d, 1e-6))) continue;
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
        }
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
    if (PATH_GRAPH_SYNC_ONLY_ON_EXPLICIT_ACTION && !state.pathGraphAllowHeavySimExport) return null;
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

  function rebuildDerivedGraphEdges(opt) {
    const forHeatmap = !!(opt && opt.forHeatmap);
    state.derivedGraphEdges = [];
    if (!state.taxiways || !state.taxiways.length) return;
    const graphSig = computeTaxiwaysGraphSig();
    let g = null;
    if (state.pathGraphCacheValid && state.pathGraphCache && state.pathGraphCacheSig === graphSig) {
      g = state.pathGraphCache;
    } else if (PATH_GRAPH_SYNC_ONLY_ON_EXPLICIT_ACTION && !forHeatmap) {
      return;
    } else {
      try {
        g = buildPathGraph(null);
      } catch (err) {
        console.error('rebuildDerivedGraphEdges: buildPathGraph failed', err);
        return;
      }
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

  function closestLayoutEdgeToPoint(wx, wy, maxDistSq) {
    if (!state.derivedGraphEdges || !state.derivedGraphEdges.length) return null;
    const click = [wx, wy];
    const lim = maxDistSq != null && isFinite(maxDistSq) ? maxDistSq : Infinity;
    let best = null;
    let bestD2 = lim;
    state.derivedGraphEdges.forEach(function(ed) {
      const pts = (ed.pts && ed.pts.length >= 2) ? ed.pts : [[ed.x1, ed.y1], [ed.x2, ed.y2]];
      for (let i = 0; i < pts.length - 1; i++) {
        const near = closestPointOnSegment(pts[i], pts[i + 1], click);
        if (!near) continue;
        const d2 = dist2(near, click);
        if (d2 < bestD2) {
          bestD2 = d2;
          best = ed;
        }
      }
    });
    return best;
  }

  /** Prefer Pro Sim ``edgeId`` on timeline samples; fall back to nearest graph edge (wider snap than picking). */
  function resolveLayoutGraphEdgeForHeatmap(a, b, mx, my, maxDistSq) {
    const raw = (b && b.edgeId != null && String(b.edgeId).trim()) ? String(b.edgeId).trim()
      : ((a && a.edgeId != null && String(a.edgeId).trim()) ? String(a.edgeId).trim() : '');
    if (raw) {
      const edges = state.derivedGraphEdges || [];
      for (let i = 0; i < edges.length; i++) {
        if (edges[i].id === raw) return edges[i];
      }
    }
    return closestLayoutEdgeToPoint(mx, my, maxDistSq);
  }

  function proSimPhaseToHeatCategory(phaseRaw) {
    const p = (phaseRaw != null && String(phaseRaw).trim()) ? String(phaseRaw).trim() : '';
    const q = p.toLowerCase().replace(/[\s-]+/g, '_');
    if (q === 'landing') return 'rotArr';
    if (q === 'arr_taxi' || q === 'arr_taxi_occupied') return 'vttArr';
    if (q === 'dep_taxi' || q === 'holding_lineup') return 'vttDep';
    if (q === 'lineup_departure') return 'rotDep';
    return null;
  }
  /** Heatmap: displayU=0 최소(네온 녹) → 1 최대(빨강), 선형 보간. displayU는 아래 왜곡 함수로부터 온다. */
  function heatmapTrafficGreenToRed(displayU) {
    const t = Math.max(0, Math.min(1, displayU));
    const r0 = 0x39, g0 = 0xff, b0 = 0x14;
    const r1 = 0xff, g1 = 0x22, b1 = 0x33;
    const r = Math.round(r0 + (r1 - r0) * t);
    const g = Math.round(g0 + (g1 - g0) * t);
    const b = Math.round(b0 + (b1 - b0) * t);
    return 'rgb(' + r + ',' + g + ',' + b + ')';
  }
  /**
   * 랭크 u∈[0,1](적게 지남→많이 지남)을 색 축으로 연속 재매핑.
   * 통과량 하위 HEATMAP_TRAFFIC_GREEN_BIAS 비율은 녹~중간(그라데이션 전반부), 나머지는 중간~빨(후반부)만 사용해 빨간 비중을 줄임.
   */
  const HEATMAP_TRAFFIC_GREEN_BIAS = 0.75;
  /** 히트맵 색·분할 기준: 월드 좌표(m) 150m 격자. 셀 단위 클립·집계. */
  const HEATMAP_GRID_STEP_M = 150;
  function heatmapTrafficRankToDisplayU(uRank) {
    const t = Math.max(0, Math.min(1, Number(uRank) || 0));
    const g = HEATMAP_TRAFFIC_GREEN_BIAS;
    if (g >= 1 - 1e-9) return t * 0.5;
    if (g <= 1e-9) return 0.5 + t * 0.5;
    if (t <= g) return (t / g) * 0.5;
    return 0.5 + ((t - g) / (1 - g)) * 0.5;
  }
  let _heatmapTrafficLegendDomSig = '';
  /** Heatmap uses full scenario [0, T]; independent of playback scrubber. */
  function heatmapClipSecFullScenarioStatic() {
    let maxT = isFinite(state.simDurationSec) ? Math.max(0, Number(state.simDurationSec)) : 0;
    (state.flights || []).forEach(function(f) {
      const tl = f && f.timeline;
      if (!tl || !tl.length) return;
      const last = tl[tl.length - 1];
      const tt = Number(last && last.t);
      if (isFinite(tt) && tt > maxT) maxT = tt;
    });
    return maxT;
  }
  /** Left legend: 5-quantile of segment-hit counts (150m cells), same colors as map. */
  function syncHeatmapTrafficLegend() {
    const root = document.getElementById('heatmap-traffic-legend');
    if (!root) return;
    const heatOk = !!state.hasSimulationResult;
    const mode = state.mapTypeMode || 'normal';
    if (!heatOk || mode !== 'heatmap') {
      _heatmapTrafficLegendDomSig = '';
      root.setAttribute('hidden', '');
      root.setAttribute('aria-hidden', 'true');
      return;
    }
    const edgeN = (state.derivedGraphEdges || []).length;
    const domSig = layoutHeatmapBakeContentSignature() + '|e' + String(edgeN);
    root.removeAttribute('hidden');
    root.removeAttribute('aria-hidden');
    const titleEl = root.querySelector('.heatmap-traffic-legend__title');
    if (titleEl) {
      titleEl.textContent = 'Segment Dwell';
    }
    if (domSig === _heatmapTrafficLegendDomSig) return;
    const rowsEl = document.getElementById('heatmapTrafficLegendRows');
    if (!rowsEl) return;
    const pack = buildHeatmapTrafficWeights();
    const gmap = pack.gridWeights || {};
    const rankU = heatmapTrafficRankUFromWeights(gmap);
    const items = [];
    Object.keys(gmap).forEach(function(k) {
      const w = gmap[k];
      if (w > 0) items.push({ id: k, w: w });
    });
    const n = items.length;
    rowsEl.innerHTML = '';
    if (n === 0) {
      const row = document.createElement('div');
      row.className = 'heatmap-traffic-legend__row heatmap-traffic-legend__row--empty';
      row.textContent = 'No heatmap data';
      rowsEl.appendChild(row);
      _heatmapTrafficLegendDomSig = domSig;
      return;
    }
    items.sort(function(a, b) {
      if (a.w !== b.w) return a.w - b.w;
      return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
    });
    const fracs = [1, 0.75, 0.5, 0.25, 0];
    const hints = ['High', '', '', '', 'Low'];
    for (let i = 0; i < fracs.length; i++) {
      const idx = Math.round(fracs[i] * (n - 1));
      const it = items[idx];
      const cnt = Math.max(0, Math.round(Number(it.w) || 0));
      const uR = rankU[it.id];
      const col = heatmapTrafficGreenToRed(heatmapTrafficRankToDisplayU(uR));
      const row = document.createElement('div');
      row.className = 'heatmap-traffic-legend__row';
      const dot = document.createElement('span');
      dot.className = 'heatmap-traffic-legend__dot';
      dot.style.background = col;
      dot.setAttribute('aria-hidden', 'true');
      const lab = document.createElement('span');
      lab.className = 'heatmap-traffic-legend__label';
      const num = document.createElement('strong');
      num.className = 'heatmap-traffic-legend__num';
      num.textContent = String(cnt);
      lab.appendChild(num);
      lab.appendChild(document.createTextNode(' hits'));
      if (hints[i]) {
        const hint = document.createElement('span');
        hint.className = 'heatmap-traffic-legend__hint';
        hint.textContent = ' · ' + hints[i];
        lab.appendChild(hint);
      }
      row.appendChild(dot);
      row.appendChild(lab);
      rowsEl.appendChild(row);
    }
    _heatmapTrafficLegendDomSig = domSig;
  }
  /**
   * 체크된 phase에서 가중치 > 0 인 항목만 모아 랭크 후 u에 매핑.
   * 최소 u=0(녹), 최대 u=1(빨); 동점은 같은 구간의 중간 u.
   */
  function heatmapTrafficRankUFromWeights(weights) {
    const rankU = Object.create(null);
    const items = [];
    Object.keys(weights).forEach(function(k) {
      const w = weights[k];
      if (w > 0) items.push({ id: k, w: w });
    });
    const n = items.length;
    if (n === 0) return rankU;
    items.sort(function(a, b) {
      if (a.w !== b.w) return a.w - b.w;
      return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
    });
    for (let i = 0; i < n; ) {
      let j = i + 1;
      while (j < n && items[j].w === items[i].w) j++;
      const mid = (i + j - 1) / 2;
      const u = n > 1 ? mid / (n - 1) : 0.5;
      for (let k = i; k < j; k++) rankU[items[k].id] = u;
      i = j;
    }
    return rankU;
  }
  /**
   * Edge·격자 셀별 세그먼트 히트: 전체 시나리오 [0, T] 안에서 타임라인 인접 샘플 한 쌍이 조건을 만족할 때마다 +1 (Δt·체류 가중 없음).
   * 재생 슬라이더와 무관하게 항상 전체 타임라인을 사용.
   */
  function buildHeatmapTrafficWeights() {
    const weights = Object.create(null);
    const gridWeights = Object.create(null);
    const edges = state.derivedGraphEdges || [];
    if (!edges.length) return { weights: weights, gridWeights: gridWeights };
    const gw = HEATMAP_GRID_STEP_M;
    const maxD2 = Math.pow(Math.max(CELL_SIZE * 2.8, 80), 2);
    const flights = state.flights || [];
    const tMax = heatmapClipSecFullScenarioStatic();
    if (tMax <= 1e-9) return { weights: weights, gridWeights: gridWeights };
    for (let fi = 0; fi < flights.length; fi++) {
      const f = flights[fi];
      if (!f) continue;
      const tl = f.timeline;
      if (!tl || tl.length < 2) continue;
      for (let i = 0; i < tl.length - 1; i++) {
        const a = tl[i];
        const b = tl[i + 1];
        if (!a || !b) continue;
        const t1 = Number(b.t);
        const t0 = Number(a.t);
        if (!isFinite(t0) || !isFinite(t1)) continue;
        const lo = Math.max(t0, 0);
        const hi = Math.min(t1, tMax);
        if (hi <= lo + 1e-9) continue;
        const ph = (a.phase != null && String(a.phase).trim()) ? String(a.phase).trim() : ((b.phase != null && String(b.phase).trim()) ? String(b.phase).trim() : 'Landing');
        const cat = proSimPhaseToHeatCategory(ph);
        if (!cat) continue;
        const mx = (Number(a.x) + Number(b.x)) * 0.5;
        const my = (Number(a.y) + Number(b.y)) * 0.5;
        if (!isFinite(mx) || !isFinite(my)) continue;
        const ed = resolveLayoutGraphEdgeForHeatmap(a, b, mx, my, maxD2);
        if (!ed || !ed.id) continue;
        const id = ed.id;
        weights[id] = (weights[id] || 0) + 1;
        const gkx = Math.floor(mx / gw);
        const gky = Math.floor(my / gw);
        const gkey = gkx + ',' + gky;
        gridWeights[gkey] = (gridWeights[gkey] || 0) + 1;
      }
    }
    return { weights: weights, gridWeights: gridWeights };
  }
  /** Liang–Barsky: 선분을 축정렬 사각형 [xmin,xmax]×[ymin,ymax] 으로 클립. */
  function clipHeatmapSegmentToAxisRect(x0, y0, x1, y1, xmin, xmax, ymin, ymax) {
    let t0 = 0, t1 = 1;
    const dx = x1 - x0, dy = y1 - y0;
    const te = 1e-10;
    function clip(p, q) {
      if (Math.abs(p) < te) return q >= -1e-8;
      const r = q / p;
      if (p < 0) {
        if (r > t1 + te) return false;
        if (r > t0) t0 = r;
      } else {
        if (r < t0 - te) return false;
        if (r < t1) t1 = r;
      }
      return true;
    }
    if (!clip(-dx, x0 - xmin)) return null;
    if (!clip(dx, xmax - x0)) return null;
    if (!clip(-dy, y0 - ymin)) return null;
    if (!clip(dy, ymax - y0)) return null;
    if (t1 < t0 - 1e-9) return null;
    return [x0 + t0 * dx, y0 + t0 * dy, x0 + t1 * dx, y0 + t1 * dy];
  }
  function heatmapPolylineBboxIntersectsRect(pts, xmin, xmax, ymin, ymax) {
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (let i = 0; i < pts.length; i++) {
      const x = pts[i][0], y = pts[i][1];
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
    return !(maxX < xmin || minX > xmax || maxY < ymin || minY > ymax);
  }
  /** 폴리라인을 한 격자 셀 안에 들어가는 연속 선분 체인들로 분해. */
  function clipHeatmapPolylineToRectChains(pts, xmin, xmax, ymin, ymax) {
    const chains = [];
    let cur = [];
    function flush() {
      if (cur.length >= 2) chains.push(cur);
      cur = [];
    }
    function appendSeg(p0x, p0y, p1x, p1y) {
      if (!cur.length) {
        cur.push([p0x, p0y]);
        cur.push([p1x, p1y]);
        return;
      }
      const L = cur[cur.length - 1];
      if (Math.hypot(L[0] - p0x, L[1] - p0y) < 1e-5) {
        if (Math.hypot(L[0] - p1x, L[1] - p1y) > 1e-5) cur.push([p1x, p1y]);
      } else {
        flush();
        cur.push([p0x, p0y]);
        cur.push([p1x, p1y]);
      }
    }
    for (let i = 0; i < pts.length - 1; i++) {
      const ax = pts[i][0], ay = pts[i][1], bx = pts[i + 1][0], by = pts[i + 1][1];
      const seg = clipHeatmapSegmentToAxisRect(ax, ay, bx, by, xmin, xmax, ymin, ymax);
      if (!seg) {
        flush();
        continue;
      }
      appendSeg(seg[0], seg[1], seg[2], seg[3]);
    }
    flush();
    return chains;
  }
  function layoutHeatmapEdgeIntersectsViewport(ed, vb) {
    if (!vb || !ed) return true;
    const pts = (ed.pts && ed.pts.length >= 2) ? ed.pts : [[ed.x1, ed.y1], [ed.x2, ed.y2]];
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (let i = 0; i < pts.length; i++) {
      const x = pts[i][0], y = pts[i][1];
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
    const pad = CELL_SIZE * 6;
    return !(maxX + pad < vb.minWx || minX - pad > vb.maxWx || maxY + pad < vb.minWy || minY - pad > vb.maxWy);
  }
  function layoutHeatmapHashStr(str) {
    const s = str == null ? '' : String(str);
    let h = 2166136261;
    for (let i = 0; i < s.length; i++) {
      h ^= s.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    return (h >>> 0).toString(36);
  }
  /** Fingerprint of timeline geometry/timing so cache invalidates when sim payload changes. */
  function layoutHeatmapFlightsDataSig() {
    const flights = state.flights || [];
    let tlPts = 0;
    let h = 2166136261;
    for (let fi = 0; fi < flights.length; fi++) {
      const f = flights[fi];
      if (!f) continue;
      const tl = f.timeline;
      if (!tl || !tl.length) continue;
      tlPts += tl.length;
      const step = Math.max(1, Math.floor(tl.length / 16));
      for (let j = 0; j < tl.length; j += step) {
        const s = tl[j];
        if (!s) continue;
        const x = Math.round(Number(s.x) * 10), y = Math.round(Number(s.y) * 10), tt = Math.round(Number(s.t) * 1000);
        h = Math.imul(h ^ x, 16777619);
        h = Math.imul(h ^ y, 16777619);
        h = Math.imul(h ^ tt, 16777619);
      }
    }
    return String(flights.length) + ':' + String(tlPts) + ':' + String(h >>> 0);
  }
  /**
   * 히트맵 SVG 재구축 조건: 맵 타입·레이아웃·항공기 지문(svg1).
   * pan/줌·tClip 제외 — 경로는 캐시, 매 프레임 matrix만 갱신.
   */
  function layoutHeatmapBakeContentSignature() {
    const mode = state.mapTypeMode || 'normal';
    if (mode === 'normal') return '';
    const graphH = layoutHeatmapHashStr(computeTaxiwaysGraphSig());
    const flightsSig = layoutHeatmapFlightsDataSig();
    return ['svg2', 'cellclip', 'staticfull', 'segcnt', 'gb' + String(HEATMAP_TRAFFIC_GREEN_BIAS), 'g' + String(HEATMAP_GRID_STEP_M), mode, graphH, flightsSig].join('|');
  }
  function ensureLayoutHeatmapSvgRefs() {
    if (!_layoutHeatmapSvg) _layoutHeatmapSvg = document.getElementById('layout-heatmap-svg');
    if (!_layoutHeatmapSvgG) _layoutHeatmapSvgG = document.getElementById('layout-heatmap-world-g');
  }
  function syncLayoutHeatmapSvgViewBox() {
    ensureLayoutHeatmapSvgRefs();
    if (!_layoutHeatmapSvg || !layoutDrawCanvas) return;
    const w = layoutDrawCanvas.width / dpr;
    const h = layoutDrawCanvas.height / dpr;
    _layoutHeatmapSvg.setAttribute('viewBox', '0 0 ' + w + ' ' + h);
  }
  function syncLayoutHeatmapSvgWorldMatrix() {
    ensureLayoutHeatmapSvgRefs();
    if (!_layoutHeatmapSvgG) return;
    const s = state.scale || 1;
    const px = state.panX;
    const py = state.panY;
    _layoutHeatmapSvgG.setAttribute('transform', 'matrix(' + s + ',0,0,' + s + ',' + px + ',' + py + ')');
  }
  function hideLayoutHeatmapSvg() {
    ensureLayoutHeatmapSvgRefs();
    if (_layoutHeatmapSvg) _layoutHeatmapSvg.style.display = 'none';
  }
  function rebuildLayoutHeatmapSvgDom() {
    ensureLayoutHeatmapSvgRefs();
    if (!_layoutHeatmapSvgG) return;
    const ns = 'http://www.w3.org/2000/svg';
    while (_layoutHeatmapSvgG.firstChild) {
      _layoutHeatmapSvgG.removeChild(_layoutHeatmapSvgG.firstChild);
    }
    const mode = state.mapTypeMode || 'normal';
    const heatmapPathStrokeWidthM = 20;
    if (mode === 'heatmap') {
      const pack = buildHeatmapTrafficWeights();
      const wmap = pack.weights;
      const gridW = pack.gridWeights || {};
      const rankG = heatmapTrafficRankUFromWeights(gridW);
      const gStep = HEATMAP_GRID_STEP_M;
      const edges = state.derivedGraphEdges || [];
      /** 동일 geometry+색 중복(겹치는 엣지 등) 제거 */
      const seenHeatmapStroke = new Set();
      Object.keys(gridW).forEach(function(gkey) {
        if (!(gridW[gkey] > 0)) return;
        const parts = gkey.split(',');
        const gx = parseInt(parts[0], 10);
        const gy = parseInt(parts[1], 10);
        if (!isFinite(gx) || !isFinite(gy)) return;
        const xmin = gx * gStep;
        const xmax = (gx + 1) * gStep;
        const ymin = gy * gStep;
        const ymax = (gy + 1) * gStep;
        const uRank = rankG[gkey] != null ? rankG[gkey] : 0;
        const col = heatmapTrafficGreenToRed(heatmapTrafficRankToDisplayU(uRank));
        for (let ei = 0; ei < edges.length; ei++) {
          const ed = edges[ei];
          if (!(wmap[ed.id] > 0)) continue;
          const pts = (ed.pts && ed.pts.length >= 2) ? ed.pts : [[ed.x1, ed.y1], [ed.x2, ed.y2]];
          if (!heatmapPolylineBboxIntersectsRect(pts, xmin, xmax, ymin, ymax)) continue;
          const chains = clipHeatmapPolylineToRectChains(pts, xmin, xmax, ymin, ymax);
          for (let ci = 0; ci < chains.length; ci++) {
            const ch = chains[ci];
            if (ch.length < 2) continue;
            let ptStr = '';
            for (let j = 0; j < ch.length; j++) {
              if (j) ptStr += ' ';
              ptStr += Number(ch[j][0]).toFixed(4) + ',' + Number(ch[j][1]).toFixed(4);
            }
            const sig = ptStr + '|' + col;
            if (seenHeatmapStroke.has(sig)) continue;
            seenHeatmapStroke.add(sig);
            const pl = document.createElementNS(ns, 'polyline');
            pl.setAttribute('points', ptStr);
            pl.setAttribute('fill', 'none');
            pl.setAttribute('stroke', col);
            pl.setAttribute('stroke-width', String(heatmapPathStrokeWidthM));
            pl.setAttribute('stroke-linecap', 'round');
            pl.setAttribute('stroke-linejoin', 'round');
            pl.setAttribute('stroke-opacity', '0.88');
            pl.setAttribute('shape-rendering', 'geometricPrecision');
            _layoutHeatmapSvgG.appendChild(pl);
          }
        }
      });
    }
  }
  function ensureDerivedGraphEdgesForHeatmap() {
    if (state.derivedGraphEdges && state.derivedGraphEdges.length) return;
    if (typeof rebuildDerivedGraphEdges === 'function') rebuildDerivedGraphEdges({ forHeatmap: true });
  }
  /** 벡터 SVG + 월드 matrix: export/오프스크린 draw 시에는 건너뜀. */
  function drawLayoutHeatmapOverlays() {
    if (layoutDrawCanvas !== canvas) return;
    if (!state.hasSimulationResult) {
      hideLayoutHeatmapSvg();
      return;
    }
    const mode = state.mapTypeMode || 'normal';
    if (mode === 'normal') {
      hideLayoutHeatmapSvg();
      return;
    }
    ensureLayoutHeatmapSvgRefs();
    if (!_layoutHeatmapSvg || !_layoutHeatmapSvgG) return;
    const contentSig = layoutHeatmapBakeContentSignature();
    if (!contentSig) return;
    if (contentSig !== _layoutHeatmapSvgContentSig) {
      ensureDerivedGraphEdgesForHeatmap();
      const ghNow = layoutHeatmapHashStr(computeTaxiwaysGraphSig());
      if (ghNow !== _layoutHeatmapBakedGraphHash || !(state.derivedGraphEdges || []).length) {
        if (typeof rebuildDerivedGraphEdges === 'function') rebuildDerivedGraphEdges({ forHeatmap: true });
        _layoutHeatmapBakedGraphHash = ghNow;
      } else {
        ensureDerivedGraphEdgesForHeatmap();
      }
      rebuildLayoutHeatmapSvgDom();
      _layoutHeatmapSvgContentSig = contentSig;
    }
    syncLayoutHeatmapSvgViewBox();
    syncLayoutHeatmapSvgWorldMatrix();
    _layoutHeatmapSvg.style.display = 'block';
    if (typeof syncHeatmapTrafficLegend === 'function') syncHeatmapTrafficLegend();
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
    if (!(state.taxiways && state.taxiways.length)) return;
    const nowMs = Date.now();
    const twN = (state.taxiways || []).length;
    const rebuildCooldownMs = twN > 500 ? 5000 : 280;
    const vb = layoutWorldViewportAabbWithBufferM(LAYOUT_RENDER_VIEWPORT_BUFFER_M);
    const scaleRef = state.scale || 1;
    const panCoarse = !!(state.isPanning && scaleRef < 0.42);
    const ultraZoomOut = scaleRef < 0.17;
    function filterPtsWorld(pts) {
      if (!pts || !pts.length) return [];
      const out = [];
      for (let i = 0; i < pts.length; i++) {
        if (worldPointInsideLayoutViewportAabb(pts[i], vb)) out.push(pts[i]);
      }
      return out;
    }
    function subsamplePts(pts) {
      const cap = ultraZoomOut ? 380 : (scaleRef < 0.3 ? 1200 : 20000);
      if (!pts || pts.length <= cap) return pts;
      const step = Math.ceil(pts.length / cap);
      const out = [];
      for (let i = 0; i < pts.length; i += step) out.push(pts[i]);
      return out;
    }
    const graphSig = computeTaxiwaysGraphSig();
    if (state.pathGraphCacheValid && state.pathGraphCacheSig && state.pathGraphCacheSig !== graphSig) {
      state.pathGraphCacheDirty = true;
      state.pathGraphInvalidatedAtMs = nowMs;
    }
    let g = null;
    const cacheHit = !!(state.pathGraphCacheValid && state.pathGraphCache && state.pathGraphCacheSig === graphSig && !state.pathGraphCache.__junctionStale);
    const hasCache = !!(state.pathGraphCacheValid && state.pathGraphCache);
    const rebuildDue = !!(state.pathGraphCacheDirty && (nowMs - (state.pathGraphInvalidatedAtMs || 0) >= rebuildCooldownMs));
    const shouldRebuildNow = (!hasCache) || rebuildDue;
    if (cacheHit) {
      g = state.pathGraphCache;
    } else if (shouldRebuildNow) {
      if (PATH_GRAPH_SYNC_ONLY_ON_EXPLICIT_ACTION) {
        g = hasCache ? state.pathGraphCache : null;
      } else if (twN >= PATH_GRAPH_ASYNC_REBUILD_MIN_TW) {
        queuePathGraphRebuild(graphSig);
        g = hasCache ? state.pathGraphCache : null;
      } else {
        try {
          g = buildPathGraph();
          state.pathGraphCache = g;
          state.pathGraphCacheValid = true;
          state.pathGraphCacheSig = graphSig;
          state.pathGraphCacheDirty = false;
        } catch (e) {
          state.pathGraphCache = null;
          state.pathGraphCacheValid = false;
          state.pathGraphCacheSig = '';
          state.pathGraphCacheDirty = true;
          console.error('drawPathJunctions: buildPathGraph failed', e);
        }
      }
    } else if (hasCache) {
      g = state.pathGraphCache;
    }
    const staleSigMismatch = !!(hasCache && state.pathGraphCacheSig && state.pathGraphCacheSig !== graphSig);
    const sigDiff = staleSigMismatch ? graphSigTaxiwayDiff(state.pathGraphCacheSig, graphSig) : { removed: [], changed: [] };
    let gGlobal = g;
    if (staleSigMismatch && sigDiff.removed.length && !PATH_GRAPH_SYNC_ONLY_ON_EXPLICIT_ACTION) gGlobal = null;
    const overlayTwIds = [];
    if (staleSigMismatch && !sigDiff.removed.length && sigDiff.changed.length) {
      const capOv = 28;
      for (let ci = 0; ci < sigDiff.changed.length && overlayTwIds.length < capOv; ci++) {
        overlayTwIds.push(sigDiff.changed[ci]);
      }
    }
    const drawId = state.taxiwayDrawingId;
    const twDraw = drawId ? (state.taxiways || []).find(function(t) { return t && String(t.id) === String(drawId); }) : null;
    const needDrawingTwOverlay = !!(twDraw && twDraw.vertices && twDraw.vertices.length >= 2);
    const apronDraftPts = getApronLinkDrawingDraftWorldPts();
    const needApronLinkDrawOverlay = !!(apronDraftPts && apronDraftPts.length >= 2);
    const dirtyApronKeys = state.apronLinkJunctionOverlayDirtyIds ? Object.keys(state.apronLinkJunctionOverlayDirtyIds) : [];
    const needApronLinkSavedOverlay = dirtyApronKeys.length > 0;
    let cacheHasDots = false;
    let redJunctions = [];
    let connectedJunctions = [];
    if (gGlobal) {
      const validJunctions = gGlobal.validJunctions || [];
      connectedJunctions = gGlobal.connectedJunctions || gGlobal.junctions || [];
      redJunctions = gGlobal.disconnectedValidJunctions != null ? gGlobal.disconnectedValidJunctions : validJunctions;
      cacheHasDots = (connectedJunctions && connectedJunctions.length > 0) || (redJunctions && redJunctions.length > 0);
    }
    if (!cacheHasDots && !needDrawingTwOverlay && !overlayTwIds.length && !needApronLinkDrawOverlay && !needApronLinkSavedOverlay) return;
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.translate(state.panX, state.panY);
    ctx.scale(state.scale, state.scale);
    const r = Math.max(4, CELL_SIZE * 0.35) * LAYOUT_VERTEX_DOT_SCALE;
    const rGreen = r * 0.7;
    if (cacheHasDots) {
      let reds = panCoarse ? [] : filterPtsWorld(redJunctions);
      let greens = filterPtsWorld(connectedJunctions);
      reds = subsamplePts(reds);
      greens = subsamplePts(greens);
      const etcMono = layerMonoEtcOn();
      ctx.fillStyle = etcMono ? C2D_LAYER_MONO_ETC_WHITE : '#ef4444';
      reds.forEach(function(p) {
        ctx.beginPath();
        ctx.arc(p[0], p[1], r, 0, Math.PI * 2);
        ctx.fill();
      });
      ctx.fillStyle = etcMono ? C2D_LAYER_MONO_ETC_WHITE : '#22c55e';
      greens.forEach(function(p) {
        ctx.beginPath();
        ctx.arc(p[0], p[1], rGreen, 0, Math.PI * 2);
        ctx.fill();
      });
    }
    /** Same threshold as `mergeNearbyPathPointsForDraw` / graph junction merge (`pathSearch.junctionMergeRadiusPx`, default 7 layout units). */
    const nearbyTol2 = Math.pow(PATH_JUNCTION_MERGE_RADIUS_PX, 2);
    const gColorRef = gGlobal || g;
    function alreadyDrawnAtOverlay(p) {
      if (!gColorRef) return false;
      const cg = gColorRef.connectedJunctions || gColorRef.junctions || [];
      const rg = gColorRef.disconnectedValidJunctions != null ? gColorRef.disconnectedValidJunctions : (gColorRef.validJunctions || []);
      let i;
      for (i = 0; i < cg.length; i++) {
        if (dist2(p, cg[i]) <= nearbyTol2) return true;
      }
      for (i = 0; i < rg.length; i++) {
        if (dist2(p, rg[i]) <= nearbyTol2) return true;
      }
      return false;
    }
    function flushOverlayDotsForTaxiway(twO) {
      if (!twO || !twO.vertices || twO.vertices.length < 2) return;
      let localPts = collectPathJunctionWorldPointsForTaxiway(twO, state.taxiways || []);
      localPts = filterPtsWorld(localPts);
      localPts = subsamplePts(localPts);
      for (let li = 0; li < localPts.length; li++) {
        const lp = localPts[li];
        if (alreadyDrawnAtOverlay(lp)) continue;
        ctx.fillStyle = overlayJunctionFillForWorldPoint(lp, gColorRef);
        ctx.beginPath();
        ctx.arc(lp[0], lp[1], rGreen, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    function flushOverlayDotsForApronLinkDraft(draftPoly) {
      if (!draftPoly || draftPoly.length < 2) return;
      let localPts = collectPathJunctionWorldPointsForOpenPolyline(draftPoly, state.taxiways || []);
      localPts = filterPtsWorld(localPts);
      localPts = subsamplePts(localPts);
      for (let li = 0; li < localPts.length; li++) {
        const lp = localPts[li];
        if (alreadyDrawnAtOverlay(lp)) continue;
        ctx.fillStyle = overlayJunctionFillForWorldPoint(lp, gColorRef);
        ctx.beginPath();
        ctx.arc(lp[0], lp[1], rGreen, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    if (needDrawingTwOverlay) flushOverlayDotsForTaxiway(twDraw);
    for (let oi = 0; oi < overlayTwIds.length; oi++) {
      const tid = overlayTwIds[oi];
      if (drawId && String(drawId) === String(tid)) continue;
      const twO = (state.taxiways || []).find(function(t) { return t && String(t.id) === String(tid); });
      flushOverlayDotsForTaxiway(twO);
    }
    if (needApronLinkDrawOverlay) flushOverlayDotsForApronLinkDraft(apronDraftPts);
    if (needApronLinkSavedOverlay && state.apronLinkJunctionOverlayDirtyIds) {
      for (let di = 0; di < dirtyApronKeys.length; di++) {
        const lid = dirtyApronKeys[di];
        const lk = (state.apronLinks || []).find(function(l) { return l && String(l.id) === String(lid); });
        if (!lk) {
          delete state.apronLinkJunctionOverlayDirtyIds[lid];
          continue;
        }
        const polySaved = getApronLinkPolylineWorldPts(lk);
        if (polySaved.length >= 2) flushOverlayDotsForApronLinkDraft(polySaved);
      }
      if (!Object.keys(state.apronLinkJunctionOverlayDirtyIds).length) state.apronLinkJunctionOverlayDirtyIds = null;
    }
    ctx.restore();
  }

  /**
   * Red X at taxiway / runway-taxiway (taxiway, runway_exit, runway_taxiway) polyline ends that meet no
   * other path vertex (within merge radius) and no apron-link vertex. Size ~ green junction dot.
   */
  function drawTaxiwayDanglingEndpointMarks() {
    if (!state.layers.junction) return;
    const list = state.taxiways || [];
    if (!list.length) return;
    const vb = layoutWorldViewportAabbWithBufferM(LAYOUT_RENDER_VIEWPORT_BUFFER_M);
    const tol2 = Math.pow(PATH_JUNCTION_MERGE_RADIUS_PX, 2);
    const r = Math.max(4, CELL_SIZE * 0.35) * LAYOUT_VERTEX_DOT_SCALE;
    const rGreen = r * 0.7;
    const armLen = rGreen * 1.15;
    function twPathPts(tw) {
      if (!tw || !tw.vertices || tw.vertices.length < 2) return null;
      return tw.vertices.map(function(v) { return cellToPixel(v.col, v.row); });
    }
    function endpointConnected(tw, ptIndex, P, pts) {
      const n = pts.length;
      const neighborIdx = ptIndex === 0 ? 1 : n - 2;
      const allTw = state.taxiways || [];
      for (let ti = 0; ti < allTw.length; ti++) {
        const tw2 = allTw[ti];
        if (!tw2) continue;
        const p2 = twPathPts(tw2);
        if (!p2) continue;
        for (let j = 0; j < p2.length; j++) {
          if (dist2(P, p2[j]) > tol2) continue;
          if (tw2 === tw && (j === ptIndex || j === neighborIdx)) continue;
          return true;
        }
      }
      const links = state.apronLinks || [];
      for (let li = 0; li < links.length; li++) {
        const poly = getApronLinkPolylineWorldPts(links[li]);
        for (let k = 0; k < poly.length; k++) {
          if (dist2(P, poly[k]) <= tol2) return true;
        }
      }
      return false;
    }
    /** 녹색 Junction(경로 그래프 connected)이 같은 위치에 있으면 기하학적 단절 X는 생략. */
    function endpointHasGreenPathJunctionNear(P) {
      const g = state.pathGraphCache;
      if (!g || g.__junctionStale) return false;
      const greens = g.connectedJunctions || g.junctions || [];
      if (!greens.length) return false;
      for (let gi = 0; gi < greens.length; gi++) {
        const q = greens[gi];
        if (!q || q.length < 2) continue;
        if (dist2(P, q) <= tol2) return true;
      }
      return false;
    }
    const seen = new Set();
    const marks = [];
    for (let ti = 0; ti < list.length; ti++) {
      const tw = list[ti];
      const ptyp = (tw && tw.pathType) || 'taxiway';
      if (ptyp !== 'taxiway' && ptyp !== 'runway_exit' && ptyp !== 'runway_taxiway') continue;
      const pts = twPathPts(tw);
      if (!pts) continue;
      const n = pts.length;
      [0, n - 1].forEach(function(ii) {
        const P = pts[ii];
        if (!worldPointInsideLayoutViewportAabb(P, vb)) return;
        if (endpointConnected(tw, ii, P, pts)) return;
        if (endpointHasGreenPathJunctionNear(P)) return;
        const key = String(Math.round(P[0] * 4)) + ',' + String(Math.round(P[1] * 4));
        if (seen.has(key)) return;
        seen.add(key);
        marks.push(P);
      });
    }
    if (!marks.length) return;
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.translate(state.panX, state.panY);
    ctx.scale(state.scale, state.scale);
    ctx.strokeStyle = layerMonoEtcOn() ? C2D_LAYER_MONO_ETC_WHITE : '#dc2626';
    ctx.lineWidth = Math.max(1.25, armLen * 0.2);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    const d = armLen * 0.92;
    marks.forEach(function(P) {
      const cx = P[0], cy = P[1];
      ctx.beginPath();
      ctx.moveTo(cx - d, cy - d);
      ctx.lineTo(cx + d, cy + d);
      ctx.moveTo(cx + d, cy - d);
      ctx.lineTo(cx - d, cy + d);
      ctx.stroke();
    });
    ctx.restore();
  }

  function drawQueueTaxiwayLaneMarkers() {
    if (!state.layers.junction) return;
    const vbQ = layoutWorldViewportAabbWithBufferM(LAYOUT_RENDER_VIEWPORT_BUFFER_M);
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.translate(state.panX, state.panY);
    ctx.scale(state.scale, state.scale);
    const r = Math.max(3.5, CELL_SIZE * 0.22) * LAYOUT_VERTEX_DOT_SCALE;
    (state.taxiways || []).forEach(function(tw) {
      if (!tw || tw.pathType !== 'general_queue_taxiway' || !tw.vertices || tw.vertices.length < 2) return;
      if (!taxiwayShouldDrawInViewport(tw, vbQ)) return;
      const jm = queueTaxiwayAutoJunctionMarkersAlong(tw, QUEUE_TAXIWAY_JUNCTION_SPACING_M);
      for (let j = 0; j < jm.length; j++) {
        const xy = jm[j].p;
        if (!worldPointInsideLayoutViewportAabb(xy, vbQ)) continue;
        ctx.beginPath();
        ctx.arc(xy[0], xy[1], r, 0, Math.PI * 2);
        ctx.fillStyle = layerMonoEtcOn() ? C2D_LAYER_MONO_ETC_WHITE : '#22c55e';
        ctx.fill();
        ctx.strokeStyle = layerMonoEtcOn() ? C2D_LAYER_MONO_ETC_WHITE : '#ef4444';
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
    Pushback: 2,
    Dep_taxi: 3,
    Holding_lineup: 4,
    Lineup_departure: 5,
  };
  function proSimPhaseStrokeStyle(phaseRaw) {
    const p = (phaseRaw != null && String(phaseRaw).trim()) ? String(phaseRaw).trim() : 'Landing';
    if (p === 'Arr_taxi_occupied') {
      return { wMul: 1.72, stroke: '#a855f7' };
    }
    if (p === 'Arr_taxi') {
      return { wMul: 1.72, stroke: '#3b82f6' };
    }
    if (p === 'Pushback') {
      return { wMul: 0.58 * 1.2, stroke: '#f97316' };
    }
    if (p === 'Dep_taxi' || p === 'Holding_lineup') {
      return { wMul: 0.58 * 1.2, stroke: '#ef4444' };
    }
    if (p === 'Lineup_departure') {
      return { wMul: 0.45 * 1.2, stroke: '#ff1493' };
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
  function proSimArrowFillForStroke(strokeHex) {
    const s = String(strokeHex || '').trim();
    const m = /^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(s);
    if (!m) return '#fafafa';
    return '#' + m[1] + m[2] + m[3];
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
    /* Base stroke: ~1.5× legacy × 1.3 (extra thickness) × 1.2 (flight schedule route reveal) */
    const baseW = Math.max(4.2, CELL_SIZE * 0.148) * 1.5 * 1.3 * 1.2;
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
    let lastDrawnPhase = null;
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
      const phaseNorm = String(phase || '').trim() || 'Landing';
      if (key && key === lastDrawnKey && phaseNorm === lastDrawnPhase) {
        return;
      }
      const st = proSimPhaseStrokeStyle(phase);
      if (phaseNorm === 'Pushback') {
        st.stroke = '#0d0d0f';
      }
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
        phase: phase,
      });
      prevEnd = edgePts[edgePts.length - 1];
      const ou = proSimOutgoingUnit(edgePts);
      if (ou.ux != null) {
        prevUx = ou.ux;
        prevUy = ou.uy;
      }
      lastDrawnKey = key;
      lastDrawnPhase = phaseNorm;
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
      const ph = String(item.phase || '').trim();
      const isRedOrPinkArrow = ph === 'Pushback' || ph === 'Dep_taxi' || ph === 'Holding_lineup' || ph === 'Lineup_departure';
      const arrowFill = isRedOrPinkArrow
        ? proSimArrowFillForStroke(item.st.stroke)
        : '#fafafa';
      const baseSpacing = Math.max(20, CELL_SIZE * 0.34) * 1.15;
      const baseHead = Math.max(4.5, CELL_SIZE * 0.135) * 1.15;
      const redPinkArrowMul = 2 * 1.5;
      const arrowSizeMul = 1.1;
      const stHex = String(item.st && item.st.stroke || '').trim().toLowerCase();
      const greenBlueArrowMul = (stHex === '#22c55e' || stHex === '#3b82f6') ? 1.3 : 1;
      let arrSp = (isRedOrPinkArrow ? baseSpacing * redPinkArrowMul : baseSpacing) * arrowSizeMul * greenBlueArrowMul;
      let arrHd = (isRedOrPinkArrow ? baseHead * redPinkArrowMul : baseHead) * arrowSizeMul * greenBlueArrowMul;
      drawProSimSegmentArrows(edgePts, arrowFill, arrSp, arrHd);
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
    if (!state.hasSimulationResult || !state.globalUpdateFresh || !state.flights.length) return;
    const vb = layoutWorldViewportAabbWithBufferM(LAYOUT_RENDER_VIEWPORT_BUFFER_M);
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.translate(state.panX, state.panY);
    ctx.scale(state.scale, state.scale);
    const tSecDraw = state.simTimeSec;
    const fcMode = state.flightColorMode || 'all';
    const fcPal = flightSimVizPaletteList();
    const fcOver = flightSimVizOverflowGray();
    const fcKeyIdx = buildFlightSim2DColorKeyIndexMap();
    state.flights.forEach(f => {
      if (flightBlockedLikeNoWay(f)) return;
      const pose = getFlightPoseAtTimeForDraw(f, tSecDraw);
      if (!pose) return;
      const x = pose.x, y = pose.y, dx = pose.dx, dy = pose.dy;
      if (!aabbIntersectsViewport(vb, { minX: x, minY: y, maxX: x, maxY: y })) return;
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
      const silhouette2D = getApronAircraftDetailedSilhouettePoints();
      const dimsM = getSimAircraftWorldDimsM(f);
      const lenM = dimsM.lenM, wingM = dimsM.wingM;
      let scaleX, scaleY, sizeRef;
      if (useDetailSil && silhouette2D.length >= 3) {
        const sp = detailedSilhouetteAxisSpans(silhouette2D);
        scaleX = lenM / sp.spanX;
        scaleY = wingM / sp.spanY;
        sizeRef = 0.5 * Math.hypot(lenM, wingM);
      } else {
        const xs = [nX, wRx, tX];
        const minXn = Math.min(xs[0], xs[1], xs[2]);
        const maxXn = Math.max(xs[0], xs[1], xs[2]);
        const lenNorm = Math.max(1e-9, maxXn - minXn);
        const wingNorm = Math.max(1e-9, uY + lY);
        scaleX = lenM / lenNorm;
        scaleY = wingM / wingNorm;
        sizeRef = 0.5 * Math.hypot(lenM, wingM);
      }
      // Pose (x,y) = front wheel (10% from nose on fuselage). Silhouette origin (0,0) is not the nose: offset draw so 10% point lands on (x,y).
      const pFwX = nX * scaleX - 0.1 * lenM;
      const drawX = x - nx * pFwX;
      const drawY = y - ny * pFwX;
      const outW = Number(_ac2d.outlineWidth);
      const outlineWidth = (isFinite(outW) && outW > 0) ? outW : 0;
      const outlineColor = _ac2d.outlineColor || '';
      const isFlightSel = state.selectedObject && state.selectedObject.type === 'flight' && state.selectedObject.id === f.id;
      const isDeadlockGhost = pose.deadlockGhost === true;
      const glyphFillCss = resolveFlightSim2DGlyphFillRgba(f, isDeadlockGhost, fcKeyIdx, fcPal, fcOver, fcMode);
      const trailGrad = simFlightTrailGradientFromFillCss(glyphFillCss);
      const preTdHalo = simPreTouchdownHaloFromFillCss(glyphFillCss);
      if (FLIGHT_TRAIL_LENGTH_M > 0 && !isFlightTrailHiddenAtSimTime(f, tSecDraw)) {
        const trailPts = getFlightTrailPolylineBackward(f, tSecDraw, FLIGHT_TRAIL_LENGTH_M);
        if (trailPts.length >= 2) {
          ctx.save();
          const x0 = trailPts[0][0], y0 = trailPts[0][1];
          const x1 = trailPts[trailPts.length - 1][0], y1 = trailPts[trailPts.length - 1][1];
          const g = ctx.createLinearGradient(x0, y0, x1, y1);
          const cFar = trailGrad.far;
          const cNearAc = trailGrad.near;
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
        ctx.fillStyle = preTdHalo.fill;
        ctx.fill();
        ctx.strokeStyle = preTdHalo.stroke;
        ctx.lineWidth = 2;
        ctx.shadowColor = preTdHalo.shadow;
        ctx.shadowBlur = c2dSimPreTouchdownHaloBlur();
        ctx.stroke();
        ctx.restore();
      }
      if (isFlightSel) {
        ctx.save();
        ctx.beginPath();
        ctx.arc(x, y, sizeRef * 0.62, 0, Math.PI * 2);
        ctx.strokeStyle = c2dFlightSelectedRingStroke();
        ctx.lineWidth = 2.5;
        ctx.shadowColor = c2dFlightSelectedRingGlow();
        ctx.shadowBlur = c2dFlightSelectedRingGlowBlur();
        ctx.stroke();
        ctx.restore();
      }
      ctx.save();
      ctx.translate(drawX, drawY);
      const ang = Math.atan2(ny, nx);
      ctx.rotate(ang);
      ctx.fillStyle = glyphFillCss;
      ctx.beginPath();
      if (useDetailSil) {
        ctx.moveTo(silhouette2D[0][0] * scaleX, silhouette2D[0][1] * scaleY);
        for (let si = 1; si < silhouette2D.length; si++) ctx.lineTo(silhouette2D[si][0] * scaleX, silhouette2D[si][1] * scaleY);
        ctx.closePath();
      } else {
        ctx.moveTo(scaleX * nX, 0);
        ctx.lineTo(scaleX * wRx, scaleY * uY);
        ctx.lineTo(scaleX * tX, 0);
        ctx.lineTo(scaleX * wRx, scaleY * lY);
        ctx.closePath();
      }
      ctx.fill();
      if (outlineWidth > 0 && outlineColor) {
        ctx.strokeStyle = isDeadlockGhost ? 'rgba(100, 116, 139, 0.55)' : outlineColor;
        ctx.lineWidth = outlineWidth;
        ctx.stroke();
      } else if (useDetailSil) {
        ctx.strokeStyle = isDeadlockGhost ? 'rgba(100, 116, 139, 0.5)' : 'rgba(15,23,42,0.85)';
        ctx.lineWidth = 1.15;
        ctx.stroke();
      }
      ctx.restore();
    });
    ctx.restore();
  }

  function ensureSimLoop() {
    if (ensureSimLoop._running) return;
    ensureSimLoop._running = true;
    ensureSimLoop._lastTs = null;
    function tick(ts) {
      let dt = 0;
      if (ensureSimLoop._lastTs != null) {
        dt = (ts - ensureSimLoop._lastTs) / 1000;
        if (dt < 0) dt = 0;
        if (dt > 0.25) dt = 0.25;
      }
      if (state.simPlaying && ensureSimLoop._playKick) {
        ensureSimLoop._playKick = false;
        dt = Math.max(dt, 1 / 60);
      }
      ensureSimLoop._lastTs = ts;
      if (state.simPlaying) {
        const lo = state.simStartSec, hi = state.simDurationSec;
        const speedRaw = state.simSpeed;
        const speed = (typeof speedRaw === 'number' && isFinite(speedRaw) && speedRaw > 0) ? speedRaw : 1;
        if (hi > lo + 1e-9) {
          state.simTimeSec = Math.min(state.simTimeSec + dt * speed, hi);
        } else {
          state.simTimeSec = lo;
        }
        const slider = document.getElementById('flightSimSlider');
        if (slider) slider.value = String(state.simTimeSec);
        updateFlightSimPlaybackLabelsDom();
        try { draw(); } catch(e) {}
        update3DSceneWhenVisible();
      }
      if (state.simPlaying) {
        window.requestAnimationFrame(tick);
      } else {
        ensureSimLoop._running = false;
        ensureSimLoop._lastTs = null;
      }
    }
    window.requestAnimationFrame(tick);
  }

  function populateAircraftSelect(sel) {
    if (!sel) return;
    const opts = AIRCRAFT_TYPES.map(a => '<option value="' + escapeHtml(String(a.id || a.name || '')) + '">' + escapeHtml(a.name || a.id || '') + '</option>').join('');
    sel.innerHTML = opts || '<option value="A320">Airbus A320</option>';
    if (!opts && sel.options.length) sel.value = 'A320';
    else if (sel.options.length) {
      let hasA320 = false;
      for (let i = 0; i < sel.options.length; i++) {
        if (sel.options[i].value === 'A320') { hasA320 = true; break; }
      }
      sel.value = hasA320 ? 'A320' : sel.options[0].value;
    }
  }
  function getAircraftConstraintOptions() {
    return AIRCRAFT_TYPES.map(function(a) {
      const id = String(a.id || a.name || '').trim();
      const label = String(a.name || a.id || id || '').trim();
      return { id: id, label: label || id };
    }).filter(function(item) { return !!item.id; });
  }
  function normalizeAllowedAircraftTypes(rawList) {
    const valid = new Set(getAircraftConstraintOptions().map(function(item) { return item.id; }));
    const out = [];
    (Array.isArray(rawList) ? rawList : []).forEach(function(item) {
      const id = String(item || '').trim();
      if (!id || !valid.has(id) || out.indexOf(id) >= 0) return;
      out.push(id);
    });
    return out;
  }
  function getStandCategoryMode(stand) {
    const isRemote = !!(stand && stand.x != null && stand.y != null && stand.x1 == null && stand.y1 == null);
    const fallback = isRemote ? (_remoteTier.defaultCategoryMode || 'aircraft') : (_pbbTier.defaultCategoryMode || 'aircraft');
    return normalizeStandCategoryMode(stand && stand.categoryMode, fallback);
  }
  function getStandAllowedAircraftTypes(stand) {
    return normalizeAllowedAircraftTypes(stand && stand.allowedAircraftTypes);
  }
  function ensureStandIcaoAndTypesCoherent(stand, defaultCategoryModeFallback) {
    if (!stand || typeof stand !== 'object') return;
    let letters = normalizeAllowedIcaoCategories(stand.allowedIcaoCategories);
    if (!letters.length) {
      const one = String(stand.category || 'C').trim().toUpperCase()[0] || 'C';
      const idx = ICAO_LETTERS_ORDER.indexOf(one);
      if (idx >= 0) letters = ICAO_LETTERS_ORDER.slice(0, idx + 1);
      else letters = ['C'];
    }
    stand.allowedIcaoCategories = letters;
    const expanded = aircraftTypeIdsForIcaoLetters(letters);
    let types = normalizeAllowedAircraftTypes(stand.allowedAircraftTypes);
    if (!types.length) {
      stand.categoryMode = 'icao';
      stand.allowedAircraftTypes = expanded;
      stand.category = representativeCategoryFromLetters(letters);
      return;
    }
    stand.categoryMode = deriveCategoryModeFromUnifiedStandPanel(types, letters);
    if (stand.categoryMode === 'icao') {
      stand.allowedAircraftTypes = expanded;
      stand.category = representativeCategoryFromLetters(letters);
    } else {
      stand.allowedAircraftTypes = types;
      if (types.length) stand.category = representativeCategoryFromAllowedTypes(types);
      else if (!stand.category) stand.category = 'C';
    }
    stand.categoryMode = normalizeStandCategoryMode(stand.categoryMode, defaultCategoryModeFallback);
  }
  function getPbbLengthMeters(pbb) {
    const x1 = Number(pbb && pbb.x1), y1 = Number(pbb && pbb.y1);
    const x2 = Number(pbb && pbb.x2), y2 = Number(pbb && pbb.y2);
    if (Number.isFinite(x1) && Number.isFinite(y1) && Number.isFinite(x2) && Number.isFinite(y2)) {
      return Math.max(1, Math.hypot(x2 - x1, y2 - y1));
    }
    const anchor = getPbbAnchorPx(pbb);
    const center = getStandConnectionPx(pbb);
    return Math.max(1, Math.hypot(center[0] - anchor[0], center[1] - anchor[1]));
  }
  function getPbbAngleDeg(pbb) {
    if (pbb && pbb.angleDeg != null) return normalizeAngleDeg(pbb.angleDeg);
    return normalizeAngleDeg(getPBBStandAngle(pbb) * 180 / Math.PI);
  }
  function getContactStandAttachedBuildingLabel(pbb) {
    const a = getPbbAnchorPx(pbb);
    const proj = getClosestTerminalEdgePoint(a[0], a[1]);
    if (proj && proj.term) return (proj.term.name || '').trim() || 'Building';
    return '—';
  }
  function getStandConnectionPx(stand) {
    if (!stand) return [0, 0];
    if (stand.apronSiteX != null && stand.apronSiteY != null) return [Number(stand.apronSiteX), Number(stand.apronSiteY)];
    if (stand.x2 != null && stand.y2 != null) return [Number(stand.x2), Number(stand.y2)];
    if (stand.x != null && stand.y != null) return [Number(stand.x), Number(stand.y)];
    return cellToPixel(stand.col || 0, stand.row || 0);
  }
  function getStandRotationHandleRadiusPx() {
    return Math.max(6, CELL_SIZE * 0.22) * LAYOUT_VERTEX_DOT_SCALE;
  }
  function getPbbRotationOriginPx(pbb) {
    return getStandConnectionPx(pbb);
  }
  function getPbbRotationHandlePx(pbb) {
    const origin = getPbbRotationOriginPx(pbb);
    const safeAngle = getPBBStandAngle(pbb);
    const cat = (pbb && pbb.category) || 'C';
    const standReach = Math.max(getStandDepthMeters(cat), getStandWidthMeters(cat)) * 0.55;
    const dist = getPbbLengthMeters(pbb) + Math.max(standReach, 10);
    return [origin[0] + Math.cos(safeAngle) * dist, origin[1] + Math.sin(safeAngle) * dist];
  }
  function getRemoteRotationHandlePx(st) {
    const center = getRemoteStandCenterPx(st);
    const angle = getRemoteStandAngleRad(st);
    const cat = (st && st.category) || 'C';
    const dist = (getStandDepthMeters(cat) / 2) + Math.max(getStandWidthMeters(cat) * 0.35, 10);
    return [center[0] + Math.cos(angle) * dist, center[1] + Math.sin(angle) * dist];
  }
  function hitTestStandRotationHandle(wx, wy) {
    const maxD2 = Math.pow(getStandRotationHandleRadiusPx() * 1.9, 2);
    if (state.selectedObject && state.selectedObject.type === 'pbb' && state.selectedObject.obj) {
      const pbb = state.selectedObject.obj;
      const handle = getPbbRotationHandlePx(pbb);
      if (dist2(handle, [wx, wy]) <= maxD2) {
        return { type: 'pbb', id: pbb.id };
      }
    }
    if (state.selectedObject && state.selectedObject.type === 'remote' && state.selectedObject.obj) {
      const st = state.selectedObject.obj;
      const handle = getRemoteRotationHandlePx(st);
      if (dist2(handle, [wx, wy]) <= maxD2) {
        return { type: 'remote', id: st.id };
      }
    }
    if (state.selectedObject && state.selectedObject.type === 'tempStand' && state.selectedObject.obj) {
      const st = state.selectedObject.obj;
      const handle = getRemoteRotationHandlePx(st);
      if (dist2(handle, [wx, wy]) <= maxD2) {
        return { type: 'tempStand', id: st.id };
      }
    }
    return null;
  }
  function drawStandRotationHandle(originPx, handlePx, active) {
    if (!originPx || !handlePx) return;
    const r = getStandRotationHandleRadiusPx();
    ctx.save();
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = active ? '#ffffff' : 'rgba(255,255,255,0.65)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(originPx[0], originPx[1]);
    ctx.lineTo(handlePx[0], handlePx[1]);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = active ? '#f43f5e' : '#a78bfa';
    ctx.beginPath();
    ctx.arc(handlePx[0], handlePx[1], r, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }
  function buildDefaultPbbBridgePoints(pbb, bridgeIndex, bridgeCount) {
    const count = Math.max(1, parseInt(bridgeCount, 10) || 1);
    const cat = (pbb && pbb.category) || 'C';
    const dep = getStandDepthMeters(cat);
    const T0x = Number(pbb.x1) || 0, T0y = Number(pbb.y1) || 0;
    const proj = getClosestTerminalEdgePoint(T0x, T0y);
    let Tx = T0x, Ty = T0y, nx = 0, ny = 1, tx = 1, ty = 0;
    if (proj && proj.term) {
      const fr = getPbbTerminalFrameFromEdge(proj.term, proj.edgeIndex, proj.point[0], proj.point[1]);
      const wx = proj.point[0], wy = proj.point[1];
      Tx = wx;
      Ty = wy;
      nx = fr.nx;
      ny = fr.ny;
      tx = fr.tx;
      ty = fr.ty;
    } else {
      const x2 = Number(pbb.x2) || 0, y2 = Number(pbb.y2) || 0;
      let nnx = x2 - Tx, nny = y2 - Ty;
      const nl = Math.hypot(nnx, nny) || 1;
      nnx /= nl;
      nny /= nl;
      nx = nnx;
      ny = nny;
      tx = -ny;
      ty = nx;
    }
    const depthM = getPbbBoardingHeightM(pbb);
    const Bx = Tx + nx * depthM, By = Ty + ny * depthM;
    const armlen = (pbb.pbbArmLenM != null && isFinite(Number(pbb.pbbArmLenM)) && Number(pbb.pbbArmLenM) > 0)
      ? Number(pbb.pbbArmLenM)
      : Math.max(dep * 0.55, 8);
    const bw = getPbbBoardingWidthM(pbb);
    const spread = Math.min(Math.max(bw * 0.38, 2.8), bw * 0.48);
    const offsetIndex = bridgeIndex - (count - 1) / 2;
    const lateral = spread * offsetIndex;
    let dirx = nx * 0.55 + tx * 0.45, diry = ny * 0.55 + ty * 0.45;
    const dl = Math.hypot(dirx, diry) || 1;
    dirx /= dl;
    diry /= dl;
    const centerApron = getStandConnectionPx(pbb);
    const endX = Bx + dirx * armlen + tx * lateral * 0.85 + (centerApron[0] - Bx) * 0.08;
    const endY = By + diry * armlen + ty * lateral * 0.85 + (centerApron[1] - By) * 0.08;
    return [
      { x: Tx, y: Ty },
      { x: Bx, y: By },
      { x: endX, y: endY },
    ];
  }
  function rebuildPbbBridgeGeometry(pbb) {
    const count = Math.max(1, Math.min(8, parseInt(pbb.pbbCount, 10) || 1));
    pbb.pbbCount = count;
    const prev = Array.isArray(pbb.pbbBridges) ? pbb.pbbBridges : [];
    pbb.pbbBridges = Array.from({ length: count }, function(_, idx) {
      const current = prev[idx];
      const points = (current && Array.isArray(current.points) && current.points.length >= 3)
        ? current.points.map(function(pt) { return { x: Number(pt.x) || 0, y: Number(pt.y) || 0 }; })
        : buildDefaultPbbBridgePoints(pbb, idx, count);
      return { id: (current && current.id) || id(), points: points };
    });
    ensurePbbBoardingWallGeometry(pbb);
    if (pbb.apronSiteX == null || pbb.apronSiteY == null) {
      const br0 = pbb.pbbBridges && pbb.pbbBridges[0];
      const p2 = br0 && br0.points && br0.points[2];
      if (p2 && isFinite(Number(p2.x)) && isFinite(Number(p2.y))) {
        pbb.apronSiteX = Number(p2.x);
        pbb.apronSiteY = Number(p2.y);
      } else {
        const ac = getStandConnectionPx(pbb);
        pbb.apronSiteX = ac[0];
        pbb.apronSiteY = ac[1];
      }
    }
  }
  function setPbbGeometryFromAngleLength(pbb, angleDeg, lengthMeters, resetBridgeGeometry) {
    pbb.angleDeg = normalizeAngleDeg(angleDeg);
    if (lengthMeters != null && lengthMeters !== undefined && Number.isFinite(Number(lengthMeters))) {
      const len = Math.max(1, Number(lengthMeters) || 1);
      pbb.pbbArmLenM = len;
      applyPbbArmLengthToBridgeEnds(pbb, len);
    }
    if (resetBridgeGeometry === true) {
      delete pbb.pbbBridges;
      rebuildPbbBridgeGeometry(pbb);
    }
  }
  function normalizeBuildingObject(termLike) {
    const term = Object.assign({}, termLike || {});
    term.buildingType = normalizeBuildingType(term.buildingType || term.terminalType);
    if (Array.isArray(term.vertices)) {
      const cs = _persistCellSizePx();
      term.vertices = term.vertices.map(function(v) {
        if (!v || typeof v !== 'object') return { col: 0, row: 0 };
        const x = Number(v.x), y = Number(v.y);
        if (isFinite(x) && isFinite(y)) return { col: x / cs, row: y / cs };
        return { col: Number(v.col) || 0, row: Number(v.row) || 0 };
      });
    }
    return term;
  }
  function normalizePbbStandObject(rawPbb) {
    const pbb = Object.assign({}, rawPbb || {});
    pbb.pbbCount = Math.max(1, Math.min(8, parseInt(pbb.pbbCount != null ? pbb.pbbCount : (_pbbTier.defaultBridgeCount || 1), 10) || 1));
    pbb.boardingWidthM = (pbb.boardingWidthM != null && isFinite(Number(pbb.boardingWidthM)) && Number(pbb.boardingWidthM) > 0)
      ? Number(pbb.boardingWidthM) : 5;
    pbb.boardingHeightM = (pbb.boardingHeightM != null && isFinite(Number(pbb.boardingHeightM)) && Number(pbb.boardingHeightM) > 0)
      ? Number(pbb.boardingHeightM) : 15;
    if (pbb.terminalContactSetbackM != null && isFinite(Number(pbb.terminalContactSetbackM)) && Number(pbb.terminalContactSetbackM) >= 0) {
      pbb.terminalContactSetbackM = Number(pbb.terminalContactSetbackM);
    } else {
      delete pbb.terminalContactSetbackM;
    }
    if (pbb.x1 != null && pbb.y1 != null && pbb.x2 != null && pbb.y2 != null) {
      if (pbb.angleDeg != null) {
        pbb.angleDeg = normalizeAngleDeg(pbb.angleDeg);
      } else if (pbb.apronSiteX != null && pbb.apronSiteY != null) {
        pbb.angleDeg = normalizeAngleDeg(Math.atan2(
          (Number(pbb.apronSiteY) || 0) - (Number(pbb.y1) || 0),
          (Number(pbb.apronSiteX) || 0) - (Number(pbb.x1) || 0)
        ) * 180 / Math.PI);
      } else {
        pbb.angleDeg = 0;
      }
      rebuildPbbBridgeGeometry(pbb);
    }
    ensureStandIcaoAndTypesCoherent(pbb, _pbbTier.defaultCategoryMode || 'aircraft');
    pbb.allowedAircraftTypes = normalizeAllowedAircraftTypes(pbb.allowedAircraftTypes);
    return pbb;
  }
  function normalizeRemoteStandObject(rawStand) {
    const stand = Object.assign({}, rawStand || {});
    stand.angleDeg = normalizeAngleDeg(stand.angleDeg != null ? stand.angleDeg : 0);
    ensureStandIcaoAndTypesCoherent(stand, _remoteTier.defaultCategoryMode || 'aircraft');
    stand.allowedAircraftTypes = normalizeAllowedAircraftTypes(stand.allowedAircraftTypes);
    return stand;
  }
  function normalizeTempStandObject(rawStand) {
    const stand = normalizeRemoteStandObject(rawStand || {});
    const x = Number(stand.x);
    const y = Number(stand.y);
    if (stand.junctionX == null || stand.junctionY == null) {
      if (Number.isFinite(x) && Number.isFinite(y)) {
        stand.junctionX = x;
        stand.junctionY = y;
      }
    } else {
      stand.junctionX = Number(stand.junctionX);
      stand.junctionY = Number(stand.junctionY);
    }
    return stand;
  }

  (function initFlightUI() {
    (function wireFlightSchedulePagerOnce() {
      if (wireFlightSchedulePagerOnce._done) return;
      wireFlightSchedulePagerOnce._done = true;
      const bPrev = document.getElementById('btnFlightSchedPrev');
      const bNext = document.getElementById('btnFlightSchedNext');
      if (!bPrev || !bNext) return;
      bPrev.addEventListener('click', function() {
        if (FLIGHT_SCHED_PAGE_SIZE <= 0 || !state.flights.length) return;
        if (state.flightSchedulePage > 0) {
          state.flightSchedulePage--;
          renderFlightList(false, false, { pageTurnOnly: true });
        }
      });
      bNext.addEventListener('click', function() {
        if (FLIGHT_SCHED_PAGE_SIZE <= 0 || !state.flights.length) return;
        const nFl = state.flights.length;
        const maxP = Math.max(0, Math.ceil(nFl / FLIGHT_SCHED_PAGE_SIZE) - 1);
        if (state.flightSchedulePage < maxP) {
          state.flightSchedulePage++;
          renderFlightList(false, false, { pageTurnOnly: true });
        }
      });
    })();
    const arrDepEl = document.getElementById('flightArrDep');
    const dwellEl = document.getElementById('flightDwell');
    const minDwellEl = document.getElementById('flightMinDwell');
    const addBtn = document.getElementById('btnAddFlight');
    const playBtn = document.getElementById('btnPlayFlights');
    const pauseBtn = document.getElementById('btnPauseFlights');
    const resetBtn = document.getElementById('btnResetFlights');
    const simSlider = document.getElementById('flightSimSlider');
    const speedSelect = document.getElementById('flightSpeed');
    const timeInputEl = document.getElementById('flightTime');
    const sibtDateInputEl = document.getElementById('flightSibtDate');
    const aircraftEl = document.getElementById('flightAircraftType');
    const regEl = document.getElementById('flightReg');
    const layoutNameInput = document.getElementById('layoutName');
    const saveLayoutBtn = document.getElementById('btnSaveLayout');
    const layoutMsgEl = document.getElementById('layoutMessage');
    const layoutLoadListEl = document.getElementById('layoutLoadList');
    const globalUpdateBtn = document.getElementById('btnGlobalUpdate');
    if (!arrDepEl) return;
    populateAircraftSelect(aircraftEl);

    function normalizeSibtDate(raw) {
      const s = (raw == null ? '' : String(raw)).trim();
      if (/^\d{4}-\d{2}-\d{2}$/.test(s)) return s;
      return DEFAULT_SIBT_DATE;
    }
    function formatFlightScheduleGateInputs(f) {
      if (!f) return;
      const tArrMin = f.timeMin != null ? f.timeMin : 0;
      if (timeInputEl) timeInputEl.value = formatMinutesToHHMMSS(tArrMin);
      if (sibtDateInputEl) {
        const dateRaw = f.sibtDate != null ? f.sibtDate : (f.serviceDate != null ? f.serviceDate : DEFAULT_SIBT_DATE);
        sibtDateInputEl.value = normalizeSibtDate(dateRaw);
      }
    }
    function bindFlightScheduleGateFieldChange(el, field) {
      if (!el) return;
      el.addEventListener('change', function() {
        if (!state.selectedObject || state.selectedObject.type !== 'flight') return;
        const f = state.selectedObject.obj;
        const m = typeof parseTimeToMinutes === 'function' ? parseTimeToMinutes(this.value || '0') : NaN;
        if (!isFinite(m)) {
          formatFlightScheduleGateInputs(f);
          return;
        }
        if (typeof applyScheduledGateTimingFromSField !== 'function') {
          formatFlightScheduleGateInputs(f);
          return;
        }
        const ok = applyScheduledGateTimingFromSField(f, field, m);
        if (!ok) {
          formatFlightScheduleGateInputs(f);
          return;
        }
        if (dwellEl) dwellEl.value = f.dwellMin != null ? f.dwellMin : 0;
        if (minDwellEl) minDwellEl.value = f.minDwellMin != null ? f.minDwellMin : 0;
        formatFlightScheduleGateInputs(f);
        if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
        const touched = f.standId ? [f.standId] : [];
        if (typeof renderFlightList === 'function')
          renderFlightList(false, false, { scheduleMode: 'incremental', dirtyFlightIds: [f.id], touchedStandIds: touched });
        if (typeof draw === 'function') draw();
      });
    }
    bindFlightScheduleGateFieldChange(timeInputEl, 'sibt');
    if (sibtDateInputEl) {
      sibtDateInputEl.addEventListener('change', function() {
        if (!state.selectedObject || state.selectedObject.type !== 'flight') return;
        const f = state.selectedObject.obj;
        f.sibtDate = normalizeSibtDate(this.value);
        f.serviceDate = f.sibtDate;
        this.value = f.sibtDate;
        if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
        const touched = f.standId ? [f.standId] : [];
        if (typeof renderFlightList === 'function')
          renderFlightList(false, false, { scheduleMode: 'incremental', dirtyFlightIds: [f.id], touchedStandIds: touched });
        if (typeof draw === 'function') draw();
      });
    }

    function randomAirlineCode() { return DEFAULT_AIRLINE_CODES[Math.floor(Math.random() * DEFAULT_AIRLINE_CODES.length)]; }
    function randomFlightNumber(airlineCode) { return (airlineCode || randomAirlineCode()) + String(Math.floor(1000 + Math.random() * 9000)); }
    function getDefaultSibtMinutes() {
      let maxT = 0;


      (state.flights || []).forEach(f => {
        if (!f) return;
        const sibt = f.sibtMin != null ? f.sibtMin : (typeof f.timeMin === 'number' ? f.timeMin : 0);
        if (isFinite(sibt) && sibt > maxT) maxT = sibt;
      });
      return maxT + 10;
    }
    if (dwellEl) {
      const syncDwell = () => {
        const isArr = arrDepEl.value === 'Arr';
        dwellEl.disabled = !isArr;
        if (!isArr) dwellEl.value = dwellEl.value || 0;
      };
      arrDepEl.addEventListener('change', syncDwell);
      syncDwell();
    }
    if (minDwellEl) {
      const syncMinDwell = () => {
        const isArr = arrDepEl.value === 'Arr';
        minDwellEl.disabled = !isArr;
        if (!isArr) minDwellEl.value = minDwellEl.value || 0;
      };
      arrDepEl.addEventListener('change', syncMinDwell);
      syncMinDwell();
    }
    const TOKEN_NODE_ORDER = ['runway','taxiway','apron','terminal'];
    function fillTokenSelects(flightCode) {
      const runwaySel = document.getElementById('tokenRunwaySelect');
      const termSel = document.getElementById('tokenTerminalSelect');
      if (runwaySel) {
        const opts = getRunwayOptions();
        runwaySel.innerHTML = '<option value="">Random</option>' + opts.map(o => '<option value="' + (o.id || '').replace(/"/g, '&quot;') + '">' + (o.name || o.id || '').replace(/</g, '&lt;') + '</option>').join('');
      }
      if (termSel) {
        const terms = (state.terminals || []).map(t => ({ id: t.id, name: (t.name || '').trim() || 'Building' }));
        termSel.innerHTML = '<option value="">Random</option>' + terms.map(o => '<option value="' + (o.id || '').replace(/"/g, '&quot;') + '">' + (o.name || o.id || '').replace(/</g, '&lt;') + '</option>').join('');
      }
    }
    function updateTokenPanesVisibility(nodes) {
      const arr = Array.isArray(nodes) ? nodes : TOKEN_NODE_ORDER;
      ['runway','taxiway','apron','terminal'].forEach((node, i) => {
        const el = document.getElementById('tokenObject' + node.charAt(0).toUpperCase() + node.slice(1));
        if (el) el.style.display = arr.indexOf(node) >= 0 ? 'block' : 'none';
      });
    }
    function proSimApiBase() {
      if (LAYOUT_API_URL && LAYOUT_API_URL !== 'null') return LAYOUT_API_URL;
      try {
        if (window.location && window.location.origin && window.location.origin !== 'null') return window.location.origin;
      } catch (e) { /* ignore */ }
      return '';
    }
    if (globalUpdateBtn) {
      globalUpdateBtn.addEventListener('click', function() {
        function failProSim(msg) {
          const m = (msg && String(msg)) || 'Pro Sim failed';
          console.error('Pro Sim:', m);
          if (typeof setGlobalUpdateProgressUi === 'function') setGlobalUpdateProgressUi(false);
          if (typeof syncProSimButtonFromDesignerPageState === 'function') syncProSimButtonFromDesignerPageState();
          if (typeof alert === 'function') alert(m);
        }
        const base = proSimApiBase();
        if (!base) {
          failProSim('Layout API가 설정되지 않았습니다. run_app.py로 서버를 띄운 뒤 다시 시도하세요.');
          return;
        }
        if (!state.designerPageUpdateFresh) {
          failProSim('먼저 Update로 경로 그래프·뷰를 동기화하세요.');
          return;
        }
        const arrRetFailRegs = typeof getArrRetFailedRegsForProSimUi === 'function' ? getArrRetFailedRegsForProSimUi() : [];
        if (arrRetFailRegs.length) {
          const n = arrRetFailRegs.length;
          const errMsg = n > 5
            ? (arrRetFailRegs.slice(0, 3).join(', ') + ', etc. — ' + n + ' aircraft failed (no valid runway exit).')
            : ('Runway exit failed: ' + arrRetFailRegs.map(function(r) { return String(r); }).join(' · '));
          failProSim(errMsg);
          return;
        }
        if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
        const playDockBtnEl = document.getElementById('btnShowPlayDock');
        if (playDockBtnEl) playDockBtnEl.disabled = true;
        try {
          if (typeof syncStateFromPanel === 'function') syncStateFromPanel();
          if (typeof syncTableToFlightState === 'function') syncTableToFlightState();
        } catch (e0) {
          failProSim(e0 && e0.message);
          return;
        }
        const layoutName = (state.currentLayoutName && String(state.currentLayoutName).trim()) || INITIAL_LAYOUT_DISPLAY_NAME || 'default_layout';
        let layoutPayload;
        let didProSimPathGraphSync = false;
        try {
          const graphSig = (typeof computeTaxiwaysGraphSig === 'function') ? computeTaxiwaysGraphSig() : '';
          const g = state.pathGraphCache;
          const needPathSync = !state.pathGraphCacheValid || !g
            || !!(g && g.__junctionStale)
            || (state.pathGraphCacheSig && state.pathGraphCacheSig !== graphSig);
          if (needPathSync && typeof applyPathGraphSyncNow === 'function') {
            applyPathGraphSyncNow();
            didProSimPathGraphSync = true;
          }
          if (didProSimPathGraphSync && typeof markDesignerPageUpdateFresh === 'function') markDesignerPageUpdateFresh();
          state.pathGraphAllowHeavySimExport = true;
          layoutPayload = serializeCurrentLayout();
        } catch (e1) {
          state.pathGraphAllowHeavySimExport = false;
          failProSim(e1 && e1.message);
          return;
        } finally {
          state.pathGraphAllowHeavySimExport = false;
        }
        let layoutForSim = layoutPayload;
        try {
          layoutForSim = JSON.parse(JSON.stringify(layoutPayload));
          delete layoutForSim.layoutMarkers;
        } catch (eStrip) {
          layoutForSim = layoutPayload;
        }
        if (typeof setGlobalUpdateProgressUi === 'function') {
          setGlobalUpdateProgressUi(true, 'Running… · 3%', 3);
        }
        fetch(base + '/api/run-simulation', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ layout: layoutForSim, layoutName: layoutName, name: layoutName }),
        }).then(function(r) {
          if (r.status === 409) {
            return r.json().then(function(d) {
              throw new Error((d && d.error) || '시뮬레이션이 이미 실행 중입니다.');
            });
          }
          if (!r.ok) {
            return r.text().then(function(t) {
              throw new Error(t || ('HTTP ' + r.status));
            });
          }
          return r.json();
        }).then(function() {
          function pollProgress() {
            fetch(base + '/api/sim-progress')
              .then(function(pr) { return pr.json(); })
              .then(function(p) {
                if (p && p.running) {
                  const pct = (p.percent != null && isFinite(Number(p.percent))) ? Number(p.percent) : 0;
                  const pctClamped = Math.max(0, Math.min(100, Math.round(pct)));
                  const runLabel = (p.runningClockLabel != null && String(p.runningClockLabel).trim() !== '')
                    ? String(p.runningClockLabel)
                    : ('Running… (' + pctClamped + '% / 00:00)');
                  if (typeof setGlobalUpdateProgressUi === 'function') {
                    setGlobalUpdateProgressUi(true, runLabel, pct);
                  }
                  setTimeout(pollProgress, 350);
                  return;
                }
                if (p && p.error) {
                  failProSim(String(p.error));
                  return;
                }
                if (typeof setGlobalUpdateProgressUi === 'function') setGlobalUpdateProgressUi(false);
                const layoutNameDone = (state.currentLayoutName && String(state.currentLayoutName).trim()) || INITIAL_LAYOUT_DISPLAY_NAME || 'default_layout';
                fetch(base + '/api/load-sim-result?name=' + encodeURIComponent(layoutNameDone))
                  .then(function(r) {
                    if (!r.ok) throw new Error('시뮬 결과를 불러오지 못했습니다.');
                    return r.json();
                  })
                  .then(function(data) {
                    if (typeof applyAirsideSimulationResultPayload === 'function') applyAirsideSimulationResultPayload(data);
                  })
                  .catch(function(e) {
                    console.warn('Pro Sim result fetch', e && e.message ? e.message : e);
                  });
              })
              .catch(function(e) {
                failProSim(e && e.message ? e.message : 'sim-progress 요청 실패');
              });
          }
          pollProgress();
        }).catch(function(e) {
          failProSim(e && e.message ? e.message : String(e));
        });
      });
    }
    const btnShowPlayDock = document.getElementById('btnShowPlayDock');
    if (btnShowPlayDock) {
      btnShowPlayDock.addEventListener('click', function() {
        state.simPlaybackDockVisible = true;
        if (typeof applySimPlaybackBarDomVisibility === 'function') applySimPlaybackBarDomVisibility();
      });
    }
    function applyTokenNodesFromCheckboxes() {
      const nodes = [];
      TOKEN_NODE_ORDER.forEach((node, i) => {
        const cb = document.getElementById('token' + node.charAt(0).toUpperCase() + node.slice(1));
        if (cb && cb.checked) nodes.push(node);
        else return;
      });
      return nodes;
    }
    function setTokenCheckboxesFromNodes(nodes) {
      const arr = Array.isArray(nodes) ? nodes : [];
      TOKEN_NODE_ORDER.forEach((node, i) => {
        const cb = document.getElementById('token' + node.charAt(0).toUpperCase() + node.slice(1));
        if (cb) cb.checked = arr.indexOf(node) >= 0;
      });
      updateTokenPanesVisibility(arr.length ? arr : TOKEN_NODE_ORDER);
    }
    ['Runway','Taxiway','Apron','Building'].forEach((name, i) => {
      const cb = document.getElementById('token' + name);
      if (!cb) return;
      cb.addEventListener('change', function() {
        if (!state.selectedObject || state.selectedObject.type !== 'flight') return;
        const f = state.selectedObject.obj;
        if (!f.token) f.token = { nodes: TOKEN_NODE_ORDER.slice(), runwayId: null, apronId: null, terminalId: null };
        if (this.checked) {
          f.token.nodes = TOKEN_NODE_ORDER.slice(0, i + 1);
          setTokenCheckboxesFromNodes(f.token.nodes);
        } else {
          f.token.nodes = TOKEN_NODE_ORDER.slice(0, i);
          setTokenCheckboxesFromNodes(f.token.nodes);
        }
        updateTokenPanesVisibility(f.token.nodes);
        rebuildSelectedFlightTimeline();
      });
    });
    const tokenRunwaySel = document.getElementById('tokenRunwaySelect');
    const tokenTerminalSel = document.getElementById('tokenTerminalSelect');
    if (tokenRunwaySel) tokenRunwaySel.addEventListener('change', function() {
      if (!state.selectedObject || state.selectedObject.type !== 'flight') return;
      const f = state.selectedObject.obj;
      if (!f.token) f.token = { nodes: TOKEN_NODE_ORDER.slice(), runwayId: null, apronId: null, terminalId: null };
      f.token.runwayId = this.value || null;
      rebuildSelectedFlightTimeline();
    });
    if (tokenTerminalSel) tokenTerminalSel.addEventListener('change', function() {
      if (!state.selectedObject || state.selectedObject.type !== 'flight') return;
      const f = state.selectedObject.obj;
      if (!f.token) f.token = { nodes: TOKEN_NODE_ORDER.slice(), runwayId: null, apronId: null, terminalId: null };
      f.token.terminalId = this.value || null;
      rebuildSelectedFlightTimeline();
    });
    const flightSubtabButtons = document.querySelectorAll('.flight-subtab');
    const flightPaneSchedule = document.getElementById('flightPaneSchedule');
    const flightPaneConfig = document.getElementById('flightPaneConfig');
    if (flightSubtabButtons && flightPaneSchedule && flightPaneConfig) {
      flightSubtabButtons.forEach(btn => {
        btn.addEventListener('click', function() {
          const target = this.getAttribute('data-flight-subtab') || 'schedule';
          flightSubtabButtons.forEach(b => b.classList.remove('active'));
          this.classList.add('active');
          if (target === 'config') {
            flightPaneSchedule.style.display = 'none';
            flightPaneConfig.style.display = 'block';
          } else {
            flightPaneSchedule.style.display = 'block';
            flightPaneConfig.style.display = 'none';
          }
        });
      });
    }
    if (addBtn) {
      addBtn.addEventListener('click', function() {
        const airlineCodeElLocal = document.getElementById('flightAirlineCode');
        const flightNumberElLocal = document.getElementById('flightFlightNumber');
        if (state.selectedObject && state.selectedObject.type === 'flight') {
          state.selectedObject = null;
          if (regEl) regEl.value = '';
          if (airlineCodeElLocal) airlineCodeElLocal.value = '';
          if (flightNumberElLocal) flightNumberElLocal.value = '';
        }
        let timeStr = (document.getElementById('flightTime').value || '').trim();
        if (!timeStr) {
          timeStr = formatMinutesToHHMMSS(DEFAULT_SIBT_TIME_MIN);
          if (timeInputEl) timeInputEl.value = timeStr;
        }
        const timeMin = parseTimeToMinutes(timeStr);
        const sibtDateForFlight = sibtDateInputEl ? normalizeSibtDate(sibtDateInputEl.value || DEFAULT_SIBT_DATE) : DEFAULT_SIBT_DATE;
        if (sibtDateInputEl) sibtDateInputEl.value = sibtDateForFlight;
        const aircraftType = (document.getElementById('flightAircraftType').value || 'A320').trim();
        const code = getCodeForAircraft(aircraftType);
        const reg = randomRegNumber();
        if (regEl) regEl.value = '';
        let airlineCode = (airlineCodeElLocal && airlineCodeElLocal.value || '').trim();
        let flightNumber = (flightNumberElLocal && flightNumberElLocal.value || '').trim();
        if (!airlineCode) airlineCode = randomAirlineCode();
        if (!flightNumber) flightNumber = randomFlightNumber(airlineCode);
        if (airlineCodeElLocal) airlineCodeElLocal.value = '';
        if (flightNumberElLocal) flightNumberElLocal.value = '';
        let dwellMin = parseFloat(document.getElementById('flightDwell').value);
        let minDwellMin = parseFloat(document.getElementById('flightMinDwell').value);
        dwellMin = (typeof dwellMin === 'number' && !isNaN(dwellMin) && dwellMin >= 0) ? dwellMin : 0;
        minDwellMin = (typeof minDwellMin === 'number' && !isNaN(minDwellMin) && minDwellMin >= 0) ? minDwellMin : 0;
        dwellMin = Math.max(SCHED_DWELL_FLOOR_MIN, dwellMin);
        minDwellMin = Math.max(SCHED_DWELL_FLOOR_MIN, minDwellMin);
        if (minDwellMin > dwellMin) minDwellMin = dwellMin;
        const arrDep = 'Arr';
        const runwayOptions = getRunwayOptions();
        const defaultRunwayId = runwayOptions.length ? (runwayOptions[0].id || null) : null;
        const defIntDomEl = document.getElementById('flightDefaultIntDom');
        const intDomNew = (defIntDomEl && String(defIntDomEl.value || '').toLowerCase() === 'dom') ? 'Dom' : 'Int';
        const f = {
          id: id(),
          arrDep,
          timeMin,
          sibtDate: sibtDateForFlight,
          serviceDate: sibtDateForFlight,
          aircraftType,
          code,
          reg,
          airlineCode,
          flightNumber,
          intDom: intDomNew,
          dwellMin,
          minDwellMin,
          arrRunwayId: defaultRunwayId,
          depRunwayId: defaultRunwayId,
          timeline: null,
          token: {
            nodes: ['runway','taxiway','apron','terminal'],
            runwayId: defaultRunwayId,
            arrRunwayId: defaultRunwayId,
            depRunwayId: defaultRunwayId,
            apronId: null,
            terminalId: null
          }
        };
        state.flights.push(f);
        if (state.hasSimulationResult && typeof recomputeSimDuration === 'function') recomputeSimDuration();
        if (typeof markProSimSyncStaleFromSchedule === 'function') markProSimSyncStaleFromSchedule();
        var addTouched = f.standId ? [f.standId] : [];
        renderFlightList(false, false, { scheduleMode: 'incremental', dirtyFlightIds: [f.id], touchedStandIds: addTouched });
        const nextDef = getDefaultSibtMinutes();
        if (timeInputEl) timeInputEl.value = formatMinutesToHHMMSS(nextDef);
        updateFlightError('');
      });
    }
    function syncFlightPanelFromSelection() {
      if (!state.selectedObject || state.selectedObject.type !== 'flight') return;
      const f = state.selectedObject.obj;
      if (arrDepEl) arrDepEl.value = 'Arr';
      if (dwellEl) {
        dwellEl.disabled = false;
        dwellEl.value = f.dwellMin || 0;
      }
      if (minDwellEl) {
        minDwellEl.disabled = false;
        minDwellEl.value = f.minDwellMin != null ? f.minDwellMin : 0;
      }
      formatFlightScheduleGateInputs(f);
      if (aircraftEl) {
        if (f.aircraftType && AIRCRAFT_BY_ID[f.aircraftType]) aircraftEl.value = f.aircraftType;
        else {
          const match = AIRCRAFT_TYPES.find(a => a.icao === (f.code || 'C'));
          aircraftEl.value = match ? match.id : (AIRCRAFT_TYPES[0] && AIRCRAFT_TYPES[0].id) || 'A320';
        }
      }
      if (regEl) regEl.value = f.reg || '';
      const airlineCodeEl = document.getElementById('flightAirlineCode');
      const flightNumberEl = document.getElementById('flightFlightNumber');
      if (airlineCodeEl) airlineCodeEl.value = f.airlineCode || '';
      if (flightNumberEl) flightNumberEl.value = f.flightNumber || '';
      if (!f.token) f.token = { nodes: TOKEN_NODE_ORDER.slice(), runwayId: null, apronId: null, terminalId: null };
      fillTokenSelects(f.code);
      setTokenCheckboxesFromNodes(f.token.nodes);
      if (tokenRunwaySel) tokenRunwaySel.value = f.token.runwayId || '';
      if (tokenTerminalSel) tokenTerminalSel.value = f.token.terminalId || '';
      if (typeof syncFlightAssignStrip === 'function') syncFlightAssignStrip();
    }
    hookSyncFlightPanelFromSelection = syncFlightPanelFromSelection;
    const origSyncPanel = syncPanelFromState;
    syncPanelFromState = function() {
      origSyncPanel();
      if (activeTab === 'flight') syncFlightPanelFromSelection();
    };
    function rebuildSelectedFlightTimeline() {
      if (!state.selectedObject || state.selectedObject.type !== 'flight') return;
      if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
      const f = state.selectedObject.obj;
      updateFlightError('');
      draw();
      var sidSched = f.standId || null;
      if (typeof renderFlightList === 'function')
        renderFlightList(false, false, { scheduleMode: 'incremental', dirtyFlightIds: [f.id], touchedStandIds: sidSched ? [sidSched] : [] });
    }
    if (arrDepEl) {
      arrDepEl.addEventListener('change', function() {
        if (!state.selectedObject || state.selectedObject.type !== 'flight') return;
        const f = state.selectedObject.obj;
        f.arrDep = this.value === 'Dep' ? 'Dep' : 'Arr';
        if (dwellEl) {
          dwellEl.disabled = f.arrDep !== 'Arr';
          if (f.arrDep !== 'Arr') {
            f.dwellMin = 0;
            dwellEl.value = 0;
          } else {
            f.dwellMin = parseFloat(dwellEl.value) || 0;
          }
        }
        if (minDwellEl) {
          minDwellEl.disabled = f.arrDep !== 'Arr';
          if (f.arrDep !== 'Arr') {
            f.minDwellMin = 0;
            minDwellEl.value = 0;
          } else {
            f.minDwellMin = Math.max(0, parseFloat(minDwellEl.value) || 0);
            minDwellEl.value = f.minDwellMin;
          }
        }
        rebuildSelectedFlightTimeline();
      });
    }
    if (aircraftEl) {
      aircraftEl.addEventListener('change', function() {
        if (!state.selectedObject || state.selectedObject.type !== 'flight') return;
        const f = state.selectedObject.obj;
        f.aircraftType = this.value || 'A320';
        f.code = getCodeForAircraft(f.aircraftType);
        rebuildSelectedFlightTimeline();
      });
    }
    if (regEl) {
      regEl.addEventListener('change', function() {
        if (!state.selectedObject || state.selectedObject.type !== 'flight') return;
        const f = state.selectedObject.obj;
        f.reg = this.value || '';
        if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
        var rs = f.standId || null;
        renderFlightList(false, false, { scheduleMode: 'incremental', dirtyFlightIds: [f.id], touchedStandIds: rs ? [rs] : [] });
        updateObjectInfo();
      });
    }
    const airlineCodeEl = document.getElementById('flightAirlineCode');
    const flightNumberEl = document.getElementById('flightFlightNumber');
    if (airlineCodeEl) {
      airlineCodeEl.addEventListener('change', function() {
        if (!state.selectedObject || state.selectedObject.type !== 'flight') return;
        const f = state.selectedObject.obj;
        f.airlineCode = this.value || '';
        if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
        var rs2 = f.standId || null;
        renderFlightList(false, false, { scheduleMode: 'incremental', dirtyFlightIds: [f.id], touchedStandIds: rs2 ? [rs2] : [] });
        updateObjectInfo();
      });
    }
    if (flightNumberEl) {
      flightNumberEl.addEventListener('change', function() {
        if (!state.selectedObject || state.selectedObject.type !== 'flight') return;
        const f = state.selectedObject.obj;
        f.flightNumber = this.value || '';
        if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
        var rs3 = f.standId || null;
        renderFlightList(false, false, { scheduleMode: 'incremental', dirtyFlightIds: [f.id], touchedStandIds: rs3 ? [rs3] : [] });
        updateObjectInfo();
      });
    }
    if (dwellEl) {
      dwellEl.addEventListener('change', function() {
        if (!state.selectedObject || state.selectedObject.type !== 'flight') return;
        const f = state.selectedObject.obj;
        let v = parseFloat(this.value);
        v = (typeof v === 'number' && !isNaN(v) && v >= 0) ? v : 0;
        let dwell = Math.max(SCHED_DWELL_FLOOR_MIN, v);
        let minDwell = f.minDwellMin != null ? f.minDwellMin : dwell;
        minDwell = Math.max(SCHED_DWELL_FLOOR_MIN, minDwell);
        if (minDwell > dwell) minDwell = dwell;
        f.dwellMin = dwell;
        f.minDwellMin = minDwell;
        this.value = f.dwellMin;
        if (minDwellEl) minDwellEl.value = f.minDwellMin;
        const tArr = f.timeMin != null ? f.timeMin : 0;
        f.sobtMin = tArr + dwell;
        f.stotMin = scheduledStotFromSobtMinutes(f, f.sobtMin);
        f.sldtMin = scheduledSldtFromSibtMinutes(f, tArr);
        if (typeof computeScheduledDisplayTimesIncremental === 'function') {
          const touched = f.standId ? [f.standId] : [];
          computeScheduledDisplayTimesIncremental(state.flights, new Set([f.id]), new Set(touched));
        }
        formatFlightScheduleGateInputs(f);
        if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
        if (typeof renderFlightList === 'function') {
          const rsD = f.standId ? [f.standId] : [];
          renderFlightList(false, false, { scheduleMode: 'incremental', dirtyFlightIds: [f.id], touchedStandIds: rsD });
        }
        rebuildSelectedFlightTimeline();
      });
    }
    if (minDwellEl) {
      minDwellEl.addEventListener('change', function() {
        if (!state.selectedObject || state.selectedObject.type !== 'flight') return;
        const f = state.selectedObject.obj;
        let dwell = f.dwellMin != null ? f.dwellMin : 0;
        dwell = Math.max(SCHED_DWELL_FLOOR_MIN, dwell);
        let v = parseFloat(this.value);
        v = (typeof v === 'number' && !isNaN(v) && v >= 0) ? v : 0;
        let minDwell = Math.max(SCHED_DWELL_FLOOR_MIN, v);
        if (minDwell > dwell) minDwell = dwell;
        f.dwellMin = dwell;
        f.minDwellMin = minDwell;
        if (dwellEl) dwellEl.value = f.dwellMin;
        this.value = f.minDwellMin;
        if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
        var rs4 = f.standId || null;
        if (typeof renderFlightList === 'function')
          renderFlightList(false, false, { scheduleMode: 'incremental', dirtyFlightIds: [f.id], touchedStandIds: rs4 ? [rs4] : [] });
      });
    }
    if (playBtn) {
      playBtn.addEventListener('click', function() {
        const errs = validateNetworkForFlights();
        if (errs.length) {
          state.simPlaying = false;
          updateFlightError(errs);
          alert('Simulation cannot be played:\\n' + errs.join('\\n'));
          return;
        }
        if (!state.flights.length) {
          updateFlightError('registered FlightThere is no.');
          alert('registered FlightThere is no.');
          return;
        }
        if (typeof recomputeSimDuration === 'function') recomputeSimDuration();
        const lo = state.simStartSec, hi = state.simDurationSec;
        let t = snapSimTimeSecForSlider(Math.max(lo, Math.min(hi, state.simTimeSec)));
        if (hi > lo && t >= hi - 1e-3) t = snapSimTimeSecForSlider(lo);
        state.simTimeSec = t;
        if (simSlider) simSlider.value = state.simTimeSec;
        state.simSliderScrubbing = false;
        if (typeof updateFlightSimPlaybackLabelsDom === 'function') updateFlightSimPlaybackLabelsDom();
        state.simPlaying = true;
        ensureSimLoop._lastTs = null;
        ensureSimLoop._playKick = true;
        ensureSimLoop();
        if (typeof syncMapTypePopoverFromState === 'function') syncMapTypePopoverFromState();
        try { draw(); } catch(e) {}
        update3DSceneWhenVisible();
      });
    }
    if (pauseBtn) {
      pauseBtn.addEventListener('click', function() {
        state.simPlaying = false;
        if (typeof ensureSimLoop === 'function') ensureSimLoop._playKick = false;
        if (typeof syncMapTypePopoverFromState === 'function') syncMapTypePopoverFromState();
        try { draw(); } catch(e) {}
        update3DSceneWhenVisible();
      });
    }
    if (resetBtn) {
      resetBtn.addEventListener('click', function() {
        state.simPlaying = false;
        if (typeof ensureSimLoop === 'function') ensureSimLoop._playKick = false;
        state.simTimeSec = snapSimTimeSecForSlider(state.simStartSec);
        if (simSlider) simSlider.value = state.simTimeSec;
        if (typeof updateFlightSimPlaybackLabelsDom === 'function') updateFlightSimPlaybackLabelsDom();
        if (typeof syncMapTypePopoverFromState === 'function') syncMapTypePopoverFromState();
        try { draw(); } catch(e) {}
        update3DSceneWhenVisible();
      });
    }
    let simSliderPointerActive = false;
    function finalizeSimSliderPointerDrag() {
      if (!simSliderPointerActive) return;
      simSliderPointerActive = false;
      state.simSliderScrubbing = false;
      if (typeof updateFlightSimPlaybackLabelsDom === 'function') updateFlightSimPlaybackLabelsDom();
      try { draw(); } catch(e) {}
      update3DSceneWhenVisible();
    }
    if (simSlider) {
      simSlider.addEventListener('pointerdown', function(e) {
        if (e.button != null && e.button !== 0) return;
        if (e.isPrimary === false) return;
        simSliderPointerActive = true;
        state.simSliderScrubbing = true;
        try { simSlider.setPointerCapture(e.pointerId); } catch (err) {}
      });
      simSlider.addEventListener('pointerup', function(e) {
        if (!simSliderPointerActive) return;
        try { simSlider.releasePointerCapture(e.pointerId); } catch (err2) {}
        finalizeSimSliderPointerDrag();
      });
      simSlider.addEventListener('pointercancel', function() {
        finalizeSimSliderPointerDrag();
      });
      simSlider.addEventListener('lostpointercapture', function() {
        finalizeSimSliderPointerDrag();
      });
      simSlider.addEventListener('input', function() {
        const secs = parseFloat(this.value);
        if (!isNaN(secs)) {
          const snapped = snapSimTimeSecForSlider(secs);
          state.simTimeSec = snapped;
          this.value = snapped;
          if (typeof updateFlightSimPlaybackLabelsDom === 'function') updateFlightSimPlaybackLabelsDom();
          if (state.simSliderScrubbing) return;
          try { draw(); } catch(e) {}
          update3DSceneWhenVisible();
        }
      });
    }
    if (speedSelect) {
      speedSelect.addEventListener('change', function() {
        const v = parseFloat(this.value);
        state.simSpeed = !isNaN(v) && v > 0 ? v : 1;
      });
      const v0 = parseFloat(speedSelect.value);
      state.simSpeed = !isNaN(v0) && v0 > 0 ? v0 : _dc.defaultSimSpeed;
    }
    const btnHideSimBar = document.getElementById('btnHideSimPlaybackBar');
    if (btnHideSimBar) {
      btnHideSimBar.addEventListener('click', function() {
        state.simPlaybackDockVisible = false;
        if (typeof applySimPlaybackBarDomVisibility === 'function') applySimPlaybackBarDomVisibility();
      });
    }
    function syncTableToFlightState() {
      const schedTable = document.querySelector('.flight-schedule-table');
      if (!schedTable || !Array.isArray(state.flights)) return;
      const rows = Array.from(schedTable.querySelectorAll('tbody tr.flight-data-row'));
      rows.forEach(function(row) {
        const fid = row.getAttribute('data-id');
        if (!fid) return;
        const f = state.flights.find(function(ff) { return ff && ff.id === fid; });
        if (!f) return;
        const tds = Array.from(row.querySelectorAll('td'));
        const k = typeof flightScheduleColumnK === 'function' ? flightScheduleColumnK() : 1;
        const etotIdx = typeof flightSchedColIndex === 'function' ? flightSchedColIndex('etot', k) : FLIGHT_SCHED_TD_ETOT;
        if (tds.length <= etotIdx) return;
        const getMin = function(idx) {
          const td = tds[idx];
          if (!td) return null;
          const dm = td.getAttribute('data-sched-min');
          if (dm != null && String(dm).trim() !== '') {
            const n = parseFloat(dm);
            return isFinite(n) ? n : null;
          }
          const txt = (td.textContent || '').trim();
          if (!txt) return null;
          const parsed = parseTimeToMinutes(txt);
          return isFinite(parsed) ? parsed : null;
        };
        const map = {
          sibtMin: typeof flightSchedColIndex === 'function' ? flightSchedColIndex('sibt', k) : FLIGHT_SCHED_TD_SIBT,
          sobtMin: typeof flightSchedColIndex === 'function' ? flightSchedColIndex('sobt', k) : FLIGHT_SCHED_TD_SOBT,
          eldtMin: typeof flightSchedColIndex === 'function' ? flightSchedColIndex('eldt', k) : FLIGHT_SCHED_TD_ELDT,
          eibtMin: typeof flightSchedColIndex === 'function' ? flightSchedColIndex('eibt', k) : FLIGHT_SCHED_TD_EIBT,
          eobtMin: typeof flightSchedColIndex === 'function' ? flightSchedColIndex('eobt', k) : FLIGHT_SCHED_TD_EOBT,
          etotMin: etotIdx
        };
        Object.keys(map).forEach(function(key) {
          const v = getMin(map[key]);
          if (v != null) f[key] = v;
        });
        if (typeof applySOffsetsFromSibtSobt === 'function') applySOffsetsFromSibtSobt(f);
      });
    }
    function setLayoutMessage(msg, isError) {
      if (!layoutMsgEl) return;
      layoutMsgEl.textContent = msg || '';
      layoutMsgEl.style.color = isError ? '#f97316' : '#9ca3af';
    }
    if (saveLayoutBtn) {
      saveLayoutBtn.addEventListener('click', function() {
        const name = (layoutNameInput && layoutNameInput.value || '').trim();
        if (!name) {
          setLayoutMessage('Please enter a save name.', true);
          return;
        }
        try {
          if (typeof syncStateFromPanel === 'function') syncStateFromPanel();
          if (typeof syncTableToFlightState === 'function') syncTableToFlightState();
          const data = serializeCurrentLayout();
          fetchSaveLayout(name, data).then(function(r) {
            if (r.ok) {
              if (typeof updateLayoutNameBar === 'function') updateLayoutNameBar(name);
              setLayoutMessage('Saved to Layout_storage as "' + name + '.json"', false);
            } else setLayoutMessage('save failed (status ' + r.status + ') — python run_app.pyAfter running with http://127.0.0.1:8501 connection', true);
          }).catch(function(e) {
            console.warn('Layout save fetch failed', e);
            setLayoutMessage('Connection failed: ' + (e && e.message) + ' — python run_app.pyAfter running with http://127.0.0.1:8501 connection', true);
          });
        } catch (e) {
          console.error(e);
          setLayoutMessage('Unable to save layout.', true);
        }
      });
    }
    function switchLayoutTab(tabId) {
      const root = document.getElementById('tab-saveload');
      if (!root) return;
      root.querySelectorAll('.layout-save-load-tab').forEach(btn => btn.classList.remove('active'));
      root.querySelectorAll('.layout-save-load-pane').forEach(p => p.classList.remove('active'));
      const btn = root.querySelector('.layout-save-load-tab[data-sltab="' + tabId + '"]');
      const pane = document.getElementById('layout-' + tabId + '-pane');
      if (btn) btn.classList.add('active');
      if (pane) pane.classList.add('active');
      if (tabId === 'load') fetchAndRefreshLayoutList();
    }
    const layoutMessageSaveEl = document.getElementById('layoutMessageSave');
    function performSaveCurrentLayout() {
      const name = (state.currentLayoutName && state.currentLayoutName.trim()) || (INITIAL_LAYOUT_DISPLAY_NAME || 'default_layout');
      try {
        if (typeof syncStateFromPanel === 'function') syncStateFromPanel();
        if (typeof syncTableToFlightState === 'function') syncTableToFlightState();
        const data = serializeCurrentLayout();
        fetchSaveLayout(name, data).then(function(r) {
          if (r.ok) {
            if (layoutMessageSaveEl) { layoutMessageSaveEl.textContent = 'saved: ' + name + '.json'; layoutMessageSaveEl.style.color = '#9ca3af'; }
            showLayoutSavedToast(name, 'success');
          } else if (layoutMessageSaveEl) {
            layoutMessageSaveEl.textContent = 'save failed (status ' + r.status + ')';
            layoutMessageSaveEl.style.color = '#f97316';
            showLayoutSavedToast(name, 'error', 'save failed (status ' + r.status + ')');
          } else {
            showLayoutSavedToast(name, 'error', 'save failed (status ' + r.status + ')');
          }
        }).catch(function(e) {
          console.warn('Object save fetch failed', e);
          if (layoutMessageSaveEl) { layoutMessageSaveEl.textContent = 'Connection failed: ' + (e && e.message); layoutMessageSaveEl.style.color = '#f97316'; }
          showLayoutSavedToast(name, 'error', 'connection failed');
        });
      } catch (e) {
        if (layoutMessageSaveEl) { layoutMessageSaveEl.textContent = 'error: ' + (e && e.message); layoutMessageSaveEl.style.color = '#f97316'; }
        showLayoutSavedToast(name, 'error', e && e.message);
      }
    }
    const btnSaveCurrent = document.getElementById('btnSaveCurrentLayout');
    if (btnSaveCurrent) btnSaveCurrent.addEventListener('click', performSaveCurrentLayout);
    window.addEventListener('keydown', function(ev) {
      const key = ev.key;
      if ((ev.ctrlKey || ev.metaKey) && !ev.altKey && (key === 's' || key === 'S')) {
        ev.preventDefault();
        try {
          const tgt = ev.target;
          if (tgt && typeof tgt.blur === 'function') {
            const tag = tgt.tagName ? String(tgt.tagName).toUpperCase() : '';
            if (tag === 'INPUT' || tag === 'TEXTAREA') tgt.blur();
          }
        } catch (_e) {}
        performSaveCurrentLayout();
      }
    });
    const saveLoadTabRoot = document.getElementById('tab-saveload');
    if (saveLoadTabRoot) {
      saveLoadTabRoot.querySelectorAll('.layout-save-load-tab[data-sltab]').forEach(btn => {
        btn.addEventListener('click', function() { switchLayoutTab(this.getAttribute('data-sltab')); });
      });
    }
    (function initLayoutLoadPaneAirportTab() {
      const pane = document.getElementById('layout-load-pane');
      if (!pane) return;
      pane.querySelectorAll('.layout-load-subtab[data-loadsub]').forEach(function(btn) {
        btn.addEventListener('click', function() {
          const sub = this.getAttribute('data-loadsub');
          pane.querySelectorAll('.layout-load-subtab[data-loadsub]').forEach(function(b) {
            const on = b.getAttribute('data-loadsub') === sub;
            b.classList.toggle('active', on);
            b.setAttribute('aria-selected', on ? 'true' : 'false');
          });
          pane.querySelectorAll('.layout-load-subpane').forEach(function(p) {
            p.classList.toggle('active', p.getAttribute('data-loadsub') === sub);
          });
          if (sub === 'layouts') fetchAndRefreshLayoutList();
        });
      });
      const btnFetch = document.getElementById('btnFetchAirportMap');
      const msgEl = document.getElementById('layoutAirportMapMsg');
      const airportListEl = document.getElementById('airportMapIcaoList');
      const airportSearchEl = document.getElementById('airportMapSearchInput');
      let selectedAirportIcao = '';
      function selectAirportIcao(icao) {
        selectedAirportIcao = String(icao || '').trim().toUpperCase();
        renderAirportRows();
      }
      function airportRowLabel(ap) {
        const left = [ap.icao, ap.iata ? ('(' + ap.iata + ')') : ''].filter(Boolean).join(' ');
        const right = [ap.name, ap.city].filter(Boolean).join(' · ');
        return left + (right ? (' — ' + right) : '');
      }
      function renderAirportRows() {
        if (!airportListEl) return;
        const q = airportSearchEl ? String(airportSearchEl.value || '').trim().toLowerCase() : '';
        let rows = AIRPORT_SEARCH_ROWS;
        if (q) {
          rows = AIRPORT_SEARCH_ROWS.filter(function(ap) {
            return ap.icao.toLowerCase().indexOf(q) >= 0
              || ap.iata.toLowerCase().indexOf(q) >= 0
              || ap.name.toLowerCase().indexOf(q) >= 0
              || ap.city.toLowerCase().indexOf(q) >= 0
              || ap.country.toLowerCase().indexOf(q) >= 0;
          });
        }
        if (!rows.length) {
          airportListEl.innerHTML = '<div style="font-size:11px;color:#9ca3af;padding:6px 2px;">No airports found.</div>';
          return;
        }
        const cap = rows.slice(0, 120);
        if (!selectedAirportIcao || cap.every(function(ap) { return ap.icao !== selectedAirportIcao; })) {
          selectedAirportIcao = cap[0].icao;
        }
        airportListEl.innerHTML = cap.map(function(ap) {
          const selected = ap.icao === selectedAirportIcao;
          const title = [ap.icao, ap.iata ? ('(' + ap.iata + ')') : ''].filter(Boolean).join(' ');
          const meta = [ap.name, ap.city, ap.country].filter(Boolean).join(' · ');
          return ''
            + '<button type="button" class="airport-option-item' + (selected ? ' selected' : '') + '" data-airport-icao="' + escapeAttr(ap.icao) + '" aria-pressed="' + (selected ? 'true' : 'false') + '">'
            +   '<span class="airport-option-title">' + escapeHtml(title) + '</span>'
            +   '<span class="airport-option-meta">' + escapeHtml(meta || airportRowLabel(ap)) + '</span>'
            + '</button>';
        }).join('');
        airportListEl.querySelectorAll('.airport-option-item[data-airport-icao]').forEach(function(btn) {
          btn.addEventListener('click', function() {
            selectAirportIcao(String(this.getAttribute('data-airport-icao') || ''));
          });
        });
      }
      if (airportSearchEl) airportSearchEl.addEventListener('input', renderAirportRows);
      if (AIRPORT_SEARCH_ROWS.length) {
        const pref = AIRPORT_SEARCH_ROWS.find(function(ap) { return ap.icao === 'RPLL'; });
        selectedAirportIcao = pref ? 'RPLL' : AIRPORT_SEARCH_ROWS[0].icao;
      }
      renderAirportRows();
      const choiceModal = document.getElementById('airportMapLoadChoiceModal');
      const choiceBodyEl = document.getElementById('airportMapLoadChoiceBody');
      const btnRedownload = document.getElementById('btnAirportMapRedownload');
      const btnUseSaved = document.getElementById('btnAirportMapUseSaved');
      let pendingAirportChoiceIcao = '';
      let pendingAirportChoiceResolve = null;
      function setAirportChoiceModalVisible(on) {
        if (!choiceModal) return;
        choiceModal.classList.toggle('is-visible', !!on);
        choiceModal.setAttribute('aria-hidden', on ? 'false' : 'true');
        if (on && btnUseSaved) {
          try {
            requestAnimationFrame(function() { btnUseSaved.focus(); });
          } catch (e) {}
        }
      }
      function finishAirportMapChoice(choice) {
        setAirportChoiceModalVisible(false);
        if (pendingAirportChoiceResolve) {
          const fn = pendingAirportChoiceResolve;
          pendingAirportChoiceResolve = null;
          fn(choice);
        }
      }
      function wireAirportChoiceModalOnce() {
        if (!choiceModal || choiceModal.dataset.wired === '1') return;
        choiceModal.dataset.wired = '1';
        if (btnRedownload) btnRedownload.addEventListener('click', function() { finishAirportMapChoice('network'); });
        if (btnUseSaved) btnUseSaved.addEventListener('click', function() { finishAirportMapChoice('saved'); });
      }
      wireAirportChoiceModalOnce();
      function applyAirportMapPayload(j, apiBase) {
        if (!j || j.ok !== true) {
          const er = (j && j.error) ? String(j.error) : 'Unknown error';
          throw new Error(er);
        }
        if (msgEl) {
          const verb = j.fromSavedMapOnly ? 'Used saved ' : 'Saved ';
          msgEl.textContent = verb + 'data/map_storage/' + (j.file || '') + ' (' + (j.featureCount != null ? j.featureCount : 0) + ' features)';
          msgEl.style.color = '#9ca3af';
        }
        if (j.layoutError) {
          console.warn('OSM → layout import failed', j.layoutError);
          if (msgEl) { msgEl.textContent += ' — layout: ' + j.layoutError; }
        }
        if (!j.layoutName || j.layoutError) return Promise.resolve(undefined);
        return fetch(apiBase + '/api/load-layout?name=' + encodeURIComponent(j.layoutName)).then(function(r2) {
          if (!r2.ok) throw new Error('load-layout HTTP ' + r2.status);
          return r2.json();
        }).then(function(layoutObj) {
          try {
            state.hasSimulationResult = false;
            applyLayoutObject(layoutObj);
            if (typeof resizeCanvas === 'function') resizeCanvas();
            if (typeof reset2DView === 'function') reset2DView();
            if (typeof syncPanelFromState === 'function') syncPanelFromState();
            if (typeof draw === 'function') draw();
            if (typeof update3DSceneWhenVisible === 'function') update3DSceneWhenVisible();
            if (typeof updateLayoutNameBar === 'function') updateLayoutNameBar(j.layoutName);
            if (typeof recomputeSimDuration === 'function') recomputeSimDuration();
            if (msgEl) { msgEl.textContent += ' · Loaded layout ' + j.layoutName; }
          } catch (err) {
            console.error('applyLayoutObject after airport fetch', err);
            throw err;
          }
        });
      }
      function postAirportMapJson(apiBase, path, icao) {
        return fetch(apiBase + path, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ icao: icao })
        })
          .then(function(r) { return r.json().then(function(j) { return { r: r, j: j }; }); })
          .then(function(o) {
            if (!o.r.ok || !o.j || o.j.ok !== true) {
              const er = (o.j && o.j.error) ? o.j.error : ('HTTP ' + o.r.status);
              throw new Error(er);
            }
            return applyAirportMapPayload(o.j, apiBase);
          });
      }
      function runAirportDownload(icao) {
        const apiBase = getLayoutApiBase();
        if (msgEl) { msgEl.textContent = 'Fetching from OpenStreetMap (Overpass)…'; msgEl.style.color = '#9ca3af'; }
        return postAirportMapJson(apiBase, '/api/fetch-airport-map', icao);
      }
      function runAirportProcessStored(icao) {
        const apiBase = getLayoutApiBase();
        if (msgEl) { msgEl.textContent = 'Rebuilding layout from saved map…'; msgEl.style.color = '#9ca3af'; }
        return postAirportMapJson(apiBase, '/api/process-stored-airport-map', icao);
      }
      if (btnFetch) {
        btnFetch.addEventListener('click', function() {
          const icao = String(selectedAirportIcao || '').trim().toUpperCase();
          if (!icao) {
            if (msgEl) { msgEl.textContent = 'Select an airport.'; msgEl.style.color = '#f97316'; }
            return;
          }
          const apiBase = getLayoutApiBase();
          btnFetch.disabled = true;
          fetch(apiBase + '/api/airport-map-exists?icao=' + encodeURIComponent(icao))
            .then(function(r) { return r.json().catch(function() { return null; }); })
            .then(function(check) {
              if (!check || check.ok !== true) return 'network';
              if (!check.exists) return 'network';
              if (!choiceModal || !choiceBodyEl) return 'network';
              pendingAirportChoiceIcao = icao;
              choiceBodyEl.innerHTML = ''
                + '<p>A saved map file already exists for <strong>' + escapeHtml(icao) + '</strong>:</p>'
                + '<p><code>data/map_storage/' + escapeHtml(check.file || (icao + '_map.json')) + '</code></p>'
                + '<p>Re-download from the source (overwrites the file), or use the saved map and rebuild the layout only.</p>';
              return new Promise(function(resolve) {
                pendingAirportChoiceResolve = resolve;
                setAirportChoiceModalVisible(true);
              });
            })
            .then(function(choice) {
              if (choice === null || choice === undefined) {
                if (msgEl) { msgEl.textContent = 'Cancelled.'; msgEl.style.color = '#9ca3af'; }
                return undefined;
              }
              if (choice === 'saved') return runAirportProcessStored(icao);
              return runAirportDownload(icao);
            })
            .catch(function(e) {
              console.warn('airport map load', e);
              if (msgEl) {
                msgEl.textContent = (e && e.message) ? String(e.message) : 'Request failed';
                msgEl.style.color = '#f97316';
              }
            })
            .finally(function() { btnFetch.disabled = false; });
        });
      }
    })();
    function getLayoutApiBase() {
      if (LAYOUT_API_URL && LAYOUT_API_URL !== 'null') return LAYOUT_API_URL;
      try { if (window.location && window.location.origin && window.location.origin !== 'null') return window.location.origin; } catch(e) {}
      return '';
    }
    function fetchSaveLayout(name, data) {
      const apiBase = (typeof getLayoutApiBase === 'function') ? getLayoutApiBase() : (LAYOUT_API_URL || '');
      return fetch(apiBase + '/api/save-layout', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ layout: data, name: name })
      });
    }
    function fetchAndRefreshLayoutList() {
      if (!layoutLoadListEl) return;
      layoutLoadListEl.innerHTML = '<div style="font-size:11px;color:#9ca3af;">Loading list...</div>';
      const apiBase = getLayoutApiBase();
      fetch(apiBase + '/api/list-layouts').then(function(r) {
        if (!r.ok) throw new Error('API Connection failed (status ' + r.status + ')');
        return r.json();
      }).then(function(data) {
        const names = (data && data.names) ? data.names : (Array.isArray(LAYOUT_NAMES) ? LAYOUT_NAMES : []);
        refreshLayoutLoadList(names);
      }).catch(function(e) {
        console.warn('Layout list fetch failed', e);
        layoutLoadListEl.innerHTML = '<div style="font-size:11px;color:#f97316;">Connection failed: ' + (e && e.message) + '</div><div style="font-size:10px;color:#9ca3af;margin-top:4px;">python run_app.py After running with http://127.0.0.1:8501 connection</div>';
      });
    }
    function refreshLayoutLoadList(namesFromApi) {
      if (!layoutLoadListEl) return;
      const names = namesFromApi != null ? (Array.isArray(namesFromApi) ? namesFromApi : []) : (Array.isArray(LAYOUT_NAMES) ? LAYOUT_NAMES : []);
      if (!names.length) {
        layoutLoadListEl.innerHTML = '<div style="font-size:11px;color:#9ca3af;">There are no saved layouts.</div>';
        return;
      }
      const reserved = { 'default_layout': true, 'current_layout': true };
      layoutLoadListEl.innerHTML = names.map(function(name) {
        const n = (name || '').replace(/"/g, '&quot;').replace(/</g, '&lt;');
        const showDel = !reserved[(name || '').toLowerCase()];
        const delBtn = showDel ? '<button type="button" class="layout-load-delete" title="Delete" data-name="' + (name || '').replace(/"/g, '&quot;') + '">×</button>' : '';
        return '<div class="layout-load-item" data-name="' + (name || '').replace(/"/g, '&quot;') + '"><span class="layout-load-name">' + n + '</span>' + delBtn + '</div>';
      }).join('');
      layoutLoadListEl.querySelectorAll('.layout-load-item').forEach(function(el) {
        const name = el.getAttribute('data-name');
        el.addEventListener('click', function(ev) {
          if (ev.target && ev.target.classList && ev.target.classList.contains('layout-load-delete')) return;
          if (!name) return;
          var apiBase = getLayoutApiBase();
          if (layoutMsgEl) { layoutMsgEl.textContent = 'Loading...'; layoutMsgEl.style.color = '#9ca3af'; }
          fetch(apiBase + '/api/load-layout?name=' + encodeURIComponent(name)).then(function(r) {
            if (!r.ok) throw new Error('not_found');
            return r.json();
          }).then(function(obj) {
            if (!obj || typeof obj !== 'object') { throw new Error('invalid_response'); }
            try {
              state.hasSimulationResult = false;
              applyLayoutObject(obj);
              resizeCanvas();
              reset2DView();
              syncPanelFromState();
              if (typeof draw === 'function') draw();
              update3DSceneWhenVisible();
              if (typeof updateLayoutNameBar === 'function') updateLayoutNameBar(name);
              if (typeof recomputeSimDuration === 'function') recomputeSimDuration();
              if (layoutMsgEl) { layoutMsgEl.textContent = 'Loaded \"' + name + '\"'; layoutMsgEl.style.color = '#9ca3af'; }
            } catch (err) {
              console.error('applyLayoutObject error', err);
              throw err;
            }
          }).catch(function(e) {
            console.warn('Layout load fetch failed', e);
            if (layoutMsgEl) { layoutMsgEl.textContent = 'Failed to load: ' + ((e && e.message) || name || '') + ' — python run_app.pyAfter running with http://127.0.0.1:8501 connection'; layoutMsgEl.style.color = '#f97316'; }
          });
        });
        el.querySelector('.layout-load-delete') && el.querySelector('.layout-load-delete').addEventListener('click', function(ev) {
          ev.stopPropagation();
          const n = this.getAttribute('data-name');
          if (!n) return;
          const apiBase = getLayoutApiBase();
          fetch(apiBase + '/api/delete-layout', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: n })
          }).then(function(r) {
            if (!r.ok) return r.json().then(function(d) { throw new Error(d.error || 'Deletion failed'); });
            return fetch(apiBase + '/api/list-layouts').then(function(r2) { return r2.json(); });
          }).then(function(data) {
            if (data && data.names) refreshLayoutLoadList(data.names);
            if (layoutMsgEl) { layoutMsgEl.textContent = 'deleted.'; layoutMsgEl.style.color = '#9ca3af'; }
          }).catch(function(e) {
            console.warn('Layout delete fetch failed', e);
            if (layoutMsgEl) { layoutMsgEl.textContent = ((e && e.message) || 'Deletion failed') + ' — python run_app.pyAfter running with http://127.0.0.1:8501 connection'; layoutMsgEl.style.color = '#f97316'; }
          });
        });
      });
    }
    fetch((getLayoutApiBase() || '') + '/api/list-layouts').then(function(r) {
      if (r.ok) return;
      var banner = document.getElementById('api-warning-banner');
      if (banner) banner.style.display = 'block';
    }).catch(function(e) {
      console.warn('API health check failed', e);
      var banner = document.getElementById('api-warning-banner');
      if (banner) banner.style.display = 'block';
    });
  })();

  document.getElementById('btnTerminalDraw').addEventListener('click', function() {
    state.selectedObject = null;
    state.selectedVertex = null;
    state.currentTerminalId = null;
    if (state.terminalDrawingId) {
      const t = state.terminals.find(x => x.id === state.terminalDrawingId);
      if (t && !t.closed && t.vertices.length >= 3) {
        t.closed = true;
        if (terminalOverlapsAnyTaxiway(t)) {
          alert('this Apron/Terminalsilver Taxiway Overlaps the center line. Please place it in a different location.');
          state.terminals = state.terminals.filter(term => term.id !== t.id);
        }
      }
      state.terminalDrawingId = null;
      state.layoutPathDrawPointer = null;
      syncPanelFromState();
      draw();
      return;
    }
    const selectedBuildingType = normalizeBuildingType(document.getElementById('buildingType') ? document.getElementById('buildingType').value : BUILDING_TYPE_DEFAULT);
    const nameBase = getDefaultBuildingNameForType(selectedBuildingType);
    const floorsEl = document.getElementById('terminalFloors');
    const f2fEl = document.getElementById('terminalFloorToFloor');
    let floors = floorsEl ? parseInt(floorsEl.value, 10) : 1;
    let f2f = f2fEl ? Number(f2fEl.value) : 4;
    floors = Math.max(1, floors || 1);
    f2f = Math.max(0.5, f2f || 4);
    const totalH = floors * f2f;
    if (findDuplicateLayoutName('terminal', null, nameBase)) {
      alertDuplicateLayoutName();
      return;
    }
    const term = { id: id(), name: nameBase, buildingType: selectedBuildingType, vertices: [], closed: false, floors, floorToFloor: f2f, floorHeight: totalH, departureCapacity: 0, arrivalCapacity: 0 };
    pushUndo();
    state.terminals.push(term);
    state.currentTerminalId = term.id;
    state.terminalDrawingId = term.id;
    syncPanelFromState();
    draw();
    if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
  });

  document.getElementById('btnTaxiwayDraw').addEventListener('click', function() {
    const hadSelection = !!state.selectedObject;
    state.selectedObject = null;
    if (state.taxiwayDrawingId) {
      const tw = state.taxiways.find(x => x.id === state.taxiwayDrawingId);
      if (tw && tw.vertices.length >= 2) {
        if (taxiwayOverlapsAnyTerminal(tw)) {
          alert('this TaxiwayIs TerminalIt overlaps with . Please draw a different path.');
          pushUndo();
          state.taxiways = state.taxiways.filter(t => t.id !== tw.id);
          if (PATH_GRAPH_SYNC_ONLY_ON_EXPLICIT_ACTION) {
            invalidatePathGraphCache(false);
            if (state.pathGraphCacheValid && state.pathGraphCache && !state.pathGraphCache.__junctionStale) {
              stripPathGraphCacheJunctionsNearTaxiwayWorld(tw);
            }
          } else {
            markPathGraphJunctionStaleShellAfterLayoutEdit();
          }
        }
        state.taxiwayDrawingId = null;
        state.layoutPathDrawPointer = null;
        syncPanelFromState();
        if (typeof redrawLayoutAfterEdit === 'function') redrawLayoutAfterEdit();
        else if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths(); else draw();
      }
      return;
    }
    const layoutMode = settingModeSelect ? settingModeSelect.value : 'taxiway';
    let pathType = pathTypeFromLayoutMode(isPathLayoutMode(layoutMode) ? layoutMode : 'taxiway');
    if (pathType === 'taxiway') {
      const kindEl = document.getElementById('taxiwayPathTypeKind');
      const kind = kindEl ? String(kindEl.value || 'normal') : 'normal';
      if (kind === 'queue') pathType = 'general_queue_taxiway';
    }
    const nameInputEl = document.getElementById('taxiwayName');
    const defaultPathName = getDefaultPathName(pathType);
    if (hadSelection && nameInputEl) nameInputEl.value = '';
    const rawName = nameInputEl ? nameInputEl.value.trim() : '';
    const nameBase = rawName || defaultPathName;
    const inputWidth = Number(document.getElementById('taxiwayWidth').value);
    const baseWidth = pathType === 'runway'
      ? RUNWAY_PATH_DEFAULT_WIDTH
      : (pathType === 'runway_exit' ? RUNWAY_EXIT_DEFAULT_WIDTH : TAXIWAY_DEFAULT_WIDTH);
    const widthVal = clampTaxiwayWidthM(pathType, inputWidth, baseWidth);
    const modeVal = (function() {
      const raw = document.getElementById('taxiwayDirectionMode') ? document.getElementById('taxiwayDirectionMode').value : '';
      if (pathType === 'runway') return (raw === 'counter_clockwise') ? 'counter_clockwise' : 'clockwise';
      return raw || 'both';
    })();
    const maxExitInput = document.getElementById('taxiwayMaxExitVel');
    const minExitInput = document.getElementById('taxiwayMinExitVel');
    const maxExitVelocity = (pathType === 'runway_exit' && maxExitInput)
      ? (function() { const mv = Number(maxExitInput.value); return isFinite(mv) && mv > 0 ? mv : null; })()
      : null;
    const minExitVelocity = (pathType === 'runway_exit' && minExitInput)
      ? (function() {
          const mv = Number(minExitInput.value);
          if (!isFinite(mv) || mv <= 0) return 15;
          if (maxExitVelocity != null && mv > maxExitVelocity) return maxExitVelocity;
          return mv;
        })()
      : undefined;
    const allowedRwDirections = (pathType === 'runway_exit')
      ? getRunwayExitAllowedDirectionsFromPanel()
      : undefined;
    const minArrVelInput = document.getElementById('runwayMinArrVelocity');
    const minArrVelocity = (pathType === 'runway' && minArrVelInput)
      ? (function() {
          const mv = Number(minArrVelInput.value);
          return (isFinite(mv) && mv > 0) ? Math.max(1, Math.min(150, mv)) : 15;
        })()
      : undefined;
    const lineupElCw = document.getElementById('runwayLineupDistM_CW');
    const lineupElCcw = document.getElementById('runwayLineupDistM_CCW');
    const lineupDistM_CW = (pathType === 'runway' && lineupElCw)
      ? (function() { const x = Number(lineupElCw.value); return (isFinite(x) && x >= 0) ? x : 0; })()
      : undefined;
    const lineupDistM_CCW = (pathType === 'runway' && lineupElCcw)
      ? (function() { const x = Number(lineupElCcw.value); return (isFinite(x) && x >= 0) ? x : 0; })()
      : undefined;
    const lineupDistM = (pathType === 'runway')
      ? ((modeVal === 'counter_clockwise') ? lineupDistM_CCW : lineupDistM_CW)
      : undefined;
    const runwayStartDispEl = document.getElementById('runwayStartDisplacedThresholdM');
    const startDisplacedThresholdM = (pathType === 'runway' && runwayStartDispEl)
      ? (function() { const x = Number(runwayStartDispEl.value); return (isFinite(x) && x >= 0) ? x : RUNWAY_START_DISPLACED_THRESHOLD_DEFAULT_M; })()
      : undefined;
    const runwayStartBlastEl = document.getElementById('runwayStartBlastPadM');
    const startBlastPadM = (pathType === 'runway' && runwayStartBlastEl)
      ? (function() { const x = Number(runwayStartBlastEl.value); return (isFinite(x) && x >= 0) ? x : RUNWAY_START_BLAST_PAD_DEFAULT_M; })()
      : undefined;
    const runwayEndDispEl = document.getElementById('runwayEndDisplacedThresholdM');
    const endDisplacedThresholdM = (pathType === 'runway' && runwayEndDispEl)
      ? (function() { const x = Number(runwayEndDispEl.value); return (isFinite(x) && x >= 0) ? x : RUNWAY_END_DISPLACED_THRESHOLD_DEFAULT_M; })()
      : undefined;
    const runwayEndBlastEl = document.getElementById('runwayEndBlastPadM');
    const endBlastPadM = (pathType === 'runway' && runwayEndBlastEl)
      ? (function() { const x = Number(runwayEndBlastEl.value); return (isFinite(x) && x >= 0) ? x : RUNWAY_END_BLAST_PAD_DEFAULT_M; })()
      : undefined;
    const taxiway = { id: id(), name: nameBase, vertices: [], width: widthVal, direction: modeVal, pathType, pavement: getPathPavementFromPanelForPathType(pathType), maxExitVelocity, minExitVelocity, allowedRwDirections, minArrVelocity, lineupDistM, lineupDistM_CW, lineupDistM_CCW, avgMoveVelocity: (function() {
      const el = document.getElementById('taxiwayAvgMoveVelocity');
      const v = el ? Number(el.value) : 10;
      return (typeof v === 'number' && isFinite(v) && v > 0) ? Math.max(1, Math.min(50, v)) : 10;
    })(), startDisplacedThresholdM, startBlastPadM, endDisplacedThresholdM, endBlastPadM };
    if (pathType !== 'runway') delete taxiway.minArrVelocity;
    if (pathType !== 'runway') delete taxiway.lineupDistM;
    if (pathType !== 'runway') delete taxiway.lineupDistM_CW;
    if (pathType !== 'runway') delete taxiway.lineupDistM_CCW;
    if (pathType !== 'runway') delete taxiway.startDisplacedThresholdM;
    if (pathType !== 'runway') delete taxiway.startBlastPadM;
    if (pathType !== 'runway') delete taxiway.endDisplacedThresholdM;
    if (pathType !== 'runway') delete taxiway.endBlastPadM;
    if (pathType !== 'runway_exit') { delete taxiway.maxExitVelocity; delete taxiway.minExitVelocity; delete taxiway.allowedRwDirections; }
    if (findDuplicateLayoutName('taxiway', null, nameBase)) {
      alertDuplicateLayoutName();
      return;
    }
    pushUndo();
    state.taxiways.push(taxiway);
    state.taxiwayDrawingId = taxiway.id;
    syncPanelFromState();
    if (typeof redrawLayoutAfterEdit === 'function') redrawLayoutAfterEdit();
    else if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths(); else draw();
  });
  const btnPbbDrawEl = document.getElementById('btnPbbDraw');
  if (btnPbbDrawEl) btnPbbDrawEl.addEventListener('click', function() {
    toggleLayoutDrawMode('pbbDrawing', 'previewPbb', null);
  });
  const btnRemoteDrawEl = document.getElementById('btnRemoteDraw');
  if (btnRemoteDrawEl) btnRemoteDrawEl.addEventListener('click', function() {
    toggleLayoutDrawMode('remoteDrawing', 'previewRemote', null);
  });
  const btnTempStandDrawEl = document.getElementById('btnTempStandDraw');
  if (btnTempStandDrawEl) btnTempStandDrawEl.addEventListener('click', function() {
    toggleLayoutDrawMode('tempStandDrawing', 'previewTempStand', null);
  });
  const btnHoldingPointDrawEl = document.getElementById('btnHoldingPointDraw');
  if (btnHoldingPointDrawEl) btnHoldingPointDrawEl.addEventListener('click', function() {
    toggleLayoutDrawMode('holdingPointDrawing', 'previewHoldingPoint', null);
  });
  const btnApronDrawEl = document.getElementById('btnApronLinkDraw');
  if (btnApronDrawEl) btnApronDrawEl.addEventListener('click', function() {
    toggleLayoutDrawMode('apronLinkDrawing', null, 'apronLinkTemp');
  });
  const btnMarkerDrawEl = document.getElementById('btnMarkerDraw');
  if (btnMarkerDrawEl) btnMarkerDrawEl.addEventListener('click', function() {
    state.markerDrawing = !state.markerDrawing;
    if (!state.markerDrawing) {
      state.markerRulerDraft = null;
      state.markerRulerHoverWorld = null;
      state.markerIslandDraft = null;
      state.markerIslandHoverWorld = null;
      state.markerAreaDraft = null;
      state.markerAreaHoverWorld = null;
      state.markerFlightHoverSnap = null;
      cancelMarkerTextDraftWithoutCommit();
    }
    syncDrawToggleButton('btnMarkerDraw', !!state.markerDrawing);
    draw();
  });
  const pathArcHudRootEl = document.getElementById('path-arc-hud');
  if (pathArcHudRootEl) {
    pathArcHudRootEl.addEventListener('mousedown', function(ev) {
      ev.stopPropagation();
    });
    pathArcHudRootEl.addEventListener('click', function(ev) {
      ev.stopPropagation();
    });
  }
  const btnPathArcToggleEl = document.getElementById('btnPathArcToggle');
  if (btnPathArcToggleEl) btnPathArcToggleEl.addEventListener('click', function(ev) {
    ev.stopPropagation();
    ev.preventDefault();
    state.pathArcModeOn = !state.pathArcModeOn;
    if (!state.pathArcModeOn && state.pathArcDrag) {
      undo();
      state.pathArcDrag = null;
    }
    updatePathArcHud();
    scheduleDraw();
  });
  const markerTextDraftInputEl = document.getElementById('markerTextDraftInput');
  if (markerTextDraftInputEl) {
    markerTextDraftInputEl.addEventListener('keydown', function(ev) {
      if (ev.key === 'Enter') {
        ev.preventDefault();
        commitMarkerTextDraft();
      }
    });
    markerTextDraftInputEl.addEventListener('blur', function() {
      if (!state.markerTextDraft || !state.markerTextDraft.active) return;
      commitMarkerTextDraft();
    });
  }
  document.querySelectorAll('.marker-tool-tab').forEach(function(tab) {
    tab.addEventListener('click', function() {
      setMarkerSubKindTab(this.getAttribute('data-marker-sub') || 'text');
      commitMarkerTextDraft();
      state.markerRulerDraft = null;
      state.markerRulerHoverWorld = null;
      state.markerIslandDraft = null;
      state.markerIslandHoverWorld = null;
      state.markerAreaDraft = null;
      state.markerAreaHoverWorld = null;
      state.markerFlightHoverSnap = null;
      scheduleDraw();
    });
  });
  populateMarkerFlightAircraftSelect();
  setMarkerSubKindTab('text');
  syncMarkerFlightAircraftRowVisibility();
  syncMarkerIslandWidthRowVisibility();
  syncMarkerNavaidRowVisibility();
  (function _wireMarkerNavaidTypeSelect() {
    const sel = document.getElementById('markerNavaidType');
    if (!sel) return;
    sel.addEventListener('change', function() {
      const so = state.selectedObject;
      if (so && so.type === 'layoutMarker' && so.obj && so.obj.kind === 'navaid') {
        so.obj.subType = getMarkerNavaidTypeFromPanel();
        if (typeof updateObjectInfo === 'function') updateObjectInfo();
        scheduleDraw();
      }
    });
  })();

  (function setupRightPanelDragResize() {
    if (!panel || !panelToggle) return;
    const rootStyle = () => getComputedStyle(document.documentElement);
    function readPxVar(name, fallback) {
      const v = parseFloat(rootStyle().getPropertyValue(name));
      return Number.isFinite(v) ? v : fallback;
    }
    function readLenVar(name, fallback) {
      const t = (rootStyle().getPropertyValue(name) || '').trim();
      return t || fallback;
    }
    function parseCssLenToPx(s, vwBase) {
      const str = String(s || '').trim().toLowerCase();
      const n = parseFloat(str);
      if (!Number.isFinite(n)) return vwBase * 0.5;
      if (str.endsWith('vw')) return (n / 100) * vwBase;
      if (str.endsWith('vh')) return (n / 100) * (typeof window !== 'undefined' ? window.innerHeight : 800);
      if (str.endsWith('%')) return (n / 100) * vwBase;
      if (str.endsWith('px')) return n;
      return n;
    }
    function maxPanelPx() {
      const m = readPxVar('--style-right-panel-resize-viewport-margin', 8);
      return Math.max(120, window.innerWidth - m);
    }
    function collapsedPx() { return readPxVar('--style-right-panel-resize-collapsed', 44); }
    function collapseBelowPx() { return readPxVar('--style-right-panel-resize-collapse-below', 96); }
    function minExpandedPx() { return readPxVar('--style-right-panel-resize-min-expanded', 220); }
    let lastExpandedWidthPx = Math.round(parseCssLenToPx(readLenVar('--style-right-panel-width-full', '50vw'), window.innerWidth));
    lastExpandedWidthPx = Math.min(maxPanelPx(), Math.max(minExpandedPx(), lastExpandedWidthPx));
    function syncToolbar(px) {
      document.documentElement.style.setProperty('--layout-toolbar-right', Math.round(px) + 'px');
    }
    function applyCollapsed() {
      panel.classList.add('collapsed');
      panel.style.width = '';
      syncToolbar(collapsedPx());
      panelToggle.textContent = '▶';
    }
    function applyExpandedWidthPx(px) {
      const cap = maxPanelPx();
      let w = Math.min(cap, Math.round(px));
      w = Math.max(minExpandedPx(), w);
      panel.classList.remove('collapsed');
      panel.style.width = w + 'px';
      lastExpandedWidthPx = w;
      syncToolbar(w);
      panelToggle.textContent = '◀';
    }
    function applyDragWidthPx(rawPx) {
      const cap = maxPanelPx();
      const c0 = collapsedPx();
      const below = collapseBelowPx();
      let w = Math.min(cap, Math.max(c0, Math.round(rawPx)));
      if (w < below) {
        panel.classList.add('collapsed');
        panel.style.width = '';
        syncToolbar(c0);
        panelToggle.textContent = '▶';
        return;
      }
      panel.classList.remove('collapsed');
      panel.style.width = w + 'px';
      syncToolbar(w);
      panelToggle.textContent = '◀';
    }
    function finishDragWidthPx(rawPx) {
      const below = collapseBelowPx();
      const cap = maxPanelPx();
      let w = Math.min(cap, Math.max(collapsedPx(), Math.round(rawPx)));
      if (w < below) {
        applyCollapsed();
        return;
      }
      w = Math.min(cap, Math.max(minExpandedPx(), w));
      applyExpandedWidthPx(w);
    }
    applyExpandedWidthPx(lastExpandedWidthPx);
    let dragStartClientX = 0;
    let dragStartWidth = 0;
    let lastMoveClientX = 0;
    let dragMoved = false;
    let resizePointerActive = false;
    let suppressToggleClick = false;
    const CLICK_MAX_MOVE = _interactionConfigNum('clickMaxMovePx', 6);
    function onResizeWindow() {
      if (panel.classList.contains('collapsed')) {
        syncToolbar(collapsedPx());
        return;
      }
      const rw = panel.getBoundingClientRect().width;
      const cap = maxPanelPx();
      if (rw > cap) applyExpandedWidthPx(cap);
      else syncToolbar(rw);
    }
    window.addEventListener('resize', onResizeWindow);
    panelToggle.addEventListener('click', function(ev) {
      if (suppressToggleClick) {
        ev.preventDefault();
        ev.stopImmediatePropagation();
        suppressToggleClick = false;
      }
    }, true);
    panelToggle.addEventListener('pointerdown', function(ev) {
      if (ev.pointerType === 'mouse' && ev.button !== 0) return;
      ev.preventDefault();
      dragMoved = false;
      resizePointerActive = true;
      dragStartClientX = ev.clientX;
      lastMoveClientX = ev.clientX;
      const c0 = collapsedPx();
      dragStartWidth = panel.classList.contains('collapsed') ? c0 : panel.getBoundingClientRect().width;
      panel.classList.add('panel-resize-dragging');
      try { panelToggle.setPointerCapture(ev.pointerId); } catch (e) {}
    });
    panelToggle.addEventListener('pointermove', function(ev) {
      if (!resizePointerActive) return;
      if (Math.abs(ev.clientX - dragStartClientX) > CLICK_MAX_MOVE) dragMoved = true;
      lastMoveClientX = ev.clientX;
      const w = dragStartWidth + (dragStartClientX - ev.clientX);
      applyDragWidthPx(w);
    });
    function endPointerDrag(ev) {
      if (!resizePointerActive) return;
      resizePointerActive = false;
      panel.classList.remove('panel-resize-dragging');
      try { if (ev && ev.pointerId != null) panelToggle.releasePointerCapture(ev.pointerId); } catch (e) {}
      if (!dragMoved) {
        if (panel.classList.contains('collapsed')) {
          applyExpandedWidthPx(lastExpandedWidthPx);
        } else {
          lastExpandedWidthPx = Math.max(minExpandedPx(), Math.min(maxPanelPx(), panel.getBoundingClientRect().width));
          applyCollapsed();
        }
        dragMoved = false;
        return;
      }
      suppressToggleClick = true;
      const endX = ev && Number.isFinite(ev.clientX) ? ev.clientX : lastMoveClientX;
      const w = dragStartWidth + (dragStartClientX - endX);
      finishDragWidthPx(w);
      dragMoved = false;
    }
    panelToggle.addEventListener('pointerup', endPointerDrag);
    panelToggle.addEventListener('pointercancel', endPointerDrag);
    panelToggle.addEventListener('lostpointercapture', function(ev) {
      if (resizePointerActive) endPointerDrag(ev);
    });
  })();

  function renderObjectList() {
    if (!objectListEl) return;
    const mode = settingModeSelect.value;
    const seen = {};
    function uniqueTitle(baseName) {
      return baseName;
    }
    const items = [];
    if (mode === 'terminal') {
      state.terminals.forEach((t, idx) => {
        if (seen['terminal_' + t.id]) return;
        seen['terminal_' + t.id] = true;
        const areaM2 = t.vertices && t.vertices.length >= 3 ? polygonAreaM2(t.vertices) : 0;
        const floors = t.floors != null ? Math.max(1, parseInt(t.floors, 10) || 1) : 1;
        const f2fRaw = t.floorToFloor != null ? Number(t.floorToFloor) : (t.floorHeight != null ? Number(t.floorHeight) : 4);
        const f2f = Math.max(0.5, f2fRaw || 4);
        const floorH = t.floorHeight != null ? Number(t.floorHeight) || (floors * f2f) : (floors * f2f);
        const dep = t.departureCapacity != null ? t.departureCapacity : 0;
        const arr = t.arrivalCapacity != null ? t.arrivalCapacity : 0;
        const baseName = (t.name && t.name.trim()) ? t.name.trim() : ('Building ' + (idx + 1));
        const buildingTheme = getBuildingTheme(t);
        items.push({
          type: 'terminal',
          id: t.id,
          title: uniqueTitle('Building | ' + baseName),
          tag: 'Height ' + floorH.toFixed(1) + ' m',
          details:
            'Type: ' + buildingTheme.label +
            '<br>' +
            'Area: ' + areaM2.toFixed(1) + ' m²' +
            '<br>Height: ' + floorH.toFixed(1) + ' m' +
            '<br>Floors: ' + floors +
            '<br>Total floor area: ' + (areaM2 * floors).toFixed(1) + ' m²' +
            '<br>Departure: ' + dep +
            '<br>Arrival: ' + arr
        });
      });
    } else if (mode === 'pbb') {
      state.pbbStands.forEach((pbb, idx) => {
        if (seen['pbb_' + pbb.id]) return;
        seen['pbb_' + pbb.id] = true;
        const baseName = (pbb.name && pbb.name.trim()) ? pbb.name.trim() : ('Contact Stand ' + (idx + 1));
        const conn = getStandConnectionPx(pbb);
        const pcol = conn[0] / CELL_SIZE;
        const prow = conn[1] / CELL_SIZE;
        const bLabel = getContactStandAttachedBuildingLabel(pbb);
        items.push({
          type: 'pbb',
          id: pbb.id,
          title: uniqueTitle('Contact Stand | ' + baseName),
          tag: 'Category ' + (pbb.category || 'C'),
          details:
            'Category: ' + (pbb.category || '—') +
            '<br>Position: (' + pcol.toFixed(1) + ',' + prow.toFixed(1) + ')' +
            '<br>Angle: ' + getPbbAngleDeg(pbb).toFixed(0) + '°' +
            '<br>Building: ' + bLabel +
            '<br>Edge cell: (' + pbb.edgeCol + ',' + pbb.edgeRow + ')'
        });
      });
    } else if (mode === 'remote') {
      state.remoteStands.forEach((st, idx) => {
        if (seen['remote_' + st.id]) return;
        seen['remote_' + st.id] = true;
        const baseName = (st.name && st.name.trim()) ? st.name.trim() : ('R' + String(idx + 1).padStart(3, '0'));
        let allowedLabel = 'All (by proximity)';
        if (Array.isArray(st.allowedTerminals) && st.allowedTerminals.length) {
          const terms = makeUniqueNamedCopy(state.terminals || [], 'name').map(function(t) { return {
            id: t.id,
            name: (t.name || '').trim() || 'Building'
          }; });
          const names = st.allowedTerminals.map(function(id) {
            const tt = terms.find(function(t) { return t.id === id; });
            return tt ? tt.name : id;
          });
          if (names.length) allowedLabel = names.join(', ');
        }
        const [rcx, rcy] = getRemoteStandCenterPx(st);
        const rcol = rcx / CELL_SIZE;
        const rrow = rcy / CELL_SIZE;
        items.push({
          type: 'remote',
          id: st.id,
          title: uniqueTitle('Remote stand | ' + baseName),
          tag: 'Category ' + (st.category || 'C'),
          details:
            'Category: ' + (st.category || '—') +
            '<br>Position: (' + rcol.toFixed(1) + ',' + rrow.toFixed(1) + ')' +
            '<br>Angle: ' + normalizeAngleDeg(st.angleDeg != null ? st.angleDeg : 0).toFixed(0) + '°' +
            '<br>available buildings: ' + allowedLabel
        });
      });
    } else if (mode === 'tempStand') {
      (state.tempStands || []).forEach(function(st, idx) {
        if (seen['tempStand_' + st.id]) return;
        seen['tempStand_' + st.id] = true;
        const baseName = (st.name && st.name.trim()) ? st.name.trim() : ('T' + String(idx + 1).padStart(3, '0'));
        let allowedLabel = 'All (by proximity)';
        if (Array.isArray(st.allowedTerminals) && st.allowedTerminals.length) {
          const terms = makeUniqueNamedCopy(state.terminals || [], 'name').map(function(t) { return {
            id: t.id,
            name: (t.name || '').trim() || 'Building'
          }; });
          const names = st.allowedTerminals.map(function(id) {
            const tt = terms.find(function(t) { return t.id === id; });
            return tt ? tt.name : id;
          });
          if (names.length) allowedLabel = names.join(', ');
        }
        const rcxcy = getRemoteStandCenterPx(st);
        const rcol = rcxcy[0] / CELL_SIZE;
        const rrow = rcxcy[1] / CELL_SIZE;
        items.push({
          type: 'tempStand',
          id: st.id,
          title: uniqueTitle('Temp stand | ' + baseName),
          tag: 'Category ' + (st.category || 'C'),
          details:
            'Category: ' + (st.category || '—') +
            '<br>Position: (' + rcol.toFixed(1) + ',' + rrow.toFixed(1) + ')' +
            '<br>Angle: ' + normalizeAngleDeg(st.angleDeg != null ? st.angleDeg : 0).toFixed(0) + '°' +
            '<br>available buildings: ' + allowedLabel +
            '<br>Taxiway centerline only'
        });
      });
    } else if (isPathLayoutMode(mode)) {
      const wantPt = pathTypeFromLayoutMode(mode);
      state.taxiways.forEach((tw, idx) => {
        if (seen['taxiway_' + tw.id]) return;
        const pt = tw.pathType || 'taxiway';
        if (wantPt === 'taxiway') {
          if (pt !== 'taxiway' && pt !== 'general_queue_taxiway') return;
        } else if (pt !== wantPt) return;
        seen['taxiway_' + tw.id] = true;
        const baseName = (tw.name && tw.name.trim()) ? tw.name.trim() : ('Taxiway ' + (idx + 1));
        const dirVal = getTaxiwayDirection(tw);
        const dirLabel = dirVal === 'clockwise' ? 'CW' : (dirVal === 'counter_clockwise' ? 'CCW' : 'Both');
        let lengthM = 0;
        if (tw.vertices && tw.vertices.length >= 2) {
          for (let i = 1; i < tw.vertices.length; i++) {
            const v0 = tw.vertices[i - 1];
            const v1 = tw.vertices[i];
            const dx = v1.col - v0.col;
            const dy = v1.row - v0.row;
            lengthM += CELL_SIZE * Math.hypot(dx, dy);
          }


        }
        const widthDefault = tw.pathType === 'runway'
          ? RUNWAY_PATH_DEFAULT_WIDTH
          : (tw.pathType === 'runway_exit' ? RUNWAY_EXIT_DEFAULT_WIDTH : TAXIWAY_DEFAULT_WIDTH);
        const widthVal = tw.width != null ? tw.width : widthDefault;
        const serTw = serializeTaxiwayWithEndpoints(tw);
        const startStr = serTw.start_point != null ? '(' + serTw.start_point.col + ',' + serTw.start_point.row + ')' : '—';
        const endStr = serTw.end_point != null ? '(' + serTw.end_point.col + ',' + serTw.end_point.row + ')' : '—';
        const heading = tw.pathType === 'runway' ? 'Runway' : (tw.pathType === 'runway_exit' ? 'Runway Taxiway' : (tw.pathType === 'apron_taxiway' ? 'Apron taxiway' : (tw.pathType === 'general_queue_taxiway' ? 'Queue taxiway' : 'Taxiway')));
        const avgVel = (typeof tw.avgMoveVelocity === 'number' && isFinite(tw.avgMoveVelocity) && tw.avgMoveVelocity > 0) ? tw.avgMoveVelocity : 10;
        const maxExit = (tw.pathType === 'runway_exit' && typeof tw.maxExitVelocity === 'number' && isFinite(tw.maxExitVelocity) && tw.maxExitVelocity > 0) ? tw.maxExitVelocity : null;
        const minExit = (tw.pathType === 'runway_exit' && typeof tw.minExitVelocity === 'number' && isFinite(tw.minExitVelocity) && tw.minExitVelocity > 0)
          ? (maxExit != null && tw.minExitVelocity > maxExit ? maxExit : tw.minExitVelocity)
          : null;
        const minArrDisplay = tw.pathType === 'runway'
          ? ((typeof tw.minArrVelocity === 'number' && isFinite(tw.minArrVelocity) && tw.minArrVelocity > 0)
            ? Math.max(1, Math.min(150, tw.minArrVelocity))
            : 15)
          : null;
        items.push({
          type: 'taxiway',
          id: tw.id,
          title: uniqueTitle(heading + ' | ' + baseName),
          tag: dirLabel,
          details:
            'Length: ' + lengthM.toFixed(0) + ' m' +
            '<br>Points: ' + tw.vertices.length +
            '<br>Width: ' + widthVal + ' m' +
            (maxExit != null ? '<br>Max exit velocity: ' + maxExit + ' m/s' : '') +
            (minExit != null ? '<br>Min exit velocity: ' + minExit + ' m/s' : '') +
            (minArrDisplay != null ? '<br>Min arr velocity: ' + minArrDisplay + ' m/s' : '') +
            (tw.pathType === 'runway'
              ? ('<br>Line up CW/CCW: ' +
                 getRunwayLineupDistMByDirection(tw, 'clockwise') + ' / ' +
                 getRunwayLineupDistMByDirection(tw, 'counter_clockwise') +
                 ' m (CW: from Start, CCW: from End)')
              : '') +
            ((tw.pathType === 'taxiway' || tw.pathType === 'apron_taxiway' || tw.pathType === 'general_queue_taxiway') ? '<br>Avg move velocity: ' + avgVel + ' m/s' : '') +
            '<br>Start point: ' + startStr +
            '<br>End point: ' + endStr
        });
      });
    } else if (mode === 'holdingPoint') {
      (state.holdingPoints || []).forEach(function(hp, idx) {
        if (!hp || seen['hp_' + hp.id]) return;
        seen['hp_' + hp.id] = true;
        const kindLabel = holdingPointKindDisplayLabel(hp.hpKind);
        const baseName = (hp.name && hp.name.trim()) ? hp.name.trim() : (kindLabel + ' ' + (idx + 1));
        const cx = Number(hp.x), cy = Number(hp.y);
        const col = cx / CELL_SIZE, row = cy / CELL_SIZE;
        const tagShort = normalizeHoldingPointKind(hp.hpKind) === 'runway_holding' ? 'RHP' : 'IHP';
        items.push({
          type: 'holdingPoint',
          id: hp.id,
          title: uniqueTitle(kindLabel + ' | ' + baseName),
          tag: tagShort + ' · ' + c2dHoldingPointDiameterM().toFixed(0) + ' m',
          details:
            'Type: ' + kindLabel +
            '<br>Position (cell): (' + col.toFixed(1) + ', ' + row.toFixed(1) + ')' +
            '<br>World: (' + cx.toFixed(0) + ', ' + cy.toFixed(0) + ')'
        });
      });
    } else if (mode === 'apronTaxiway') {
      state.apronLinks.forEach((lk, idx) => {
        if (seen['apron_' + lk.id]) return;
        seen['apron_' + lk.id] = true;
        const stand = findStandById(lk.pbbId);
        const tw = state.taxiways.find(t => t.id === lk.taxiwayId);
        const title = getApronLinkDisplayName(lk);
        const standLabel = stand && stand.name ? stand.name : lk.pbbId;
        const details = 'Stand: ' + standLabel +
          ', Taxiway: ' + (tw && tw.name ? tw.name : lk.taxiwayId);
        items.push({
          type: 'apronLink',
          id: lk.id,
          title: uniqueTitle('Apron–Taxiway | ' + title),
          tag: 'Apron–Taxiway',
          details
        });
      });
    } else if (mode === 'edge') {
      rebuildDerivedGraphEdges();
      (state.derivedGraphEdges || []).forEach(function(ed) {
        items.push({
          type: 'layoutEdge',
          id: ed.id,
          title: 'Edge | ' + getLayoutEdgeDisplayName(ed),
          tag: 'Graph',
          details:
            'Length (graph): ' + Math.round(ed.dist) +
            '<br>Pixel span: (' + ed.x1.toFixed(0) + ', ' + ed.y1.toFixed(0) + ') → (' + ed.x2.toFixed(0) + ', ' + ed.y2.toFixed(0) + ')' +
            '<br>Polyline points: ' + ((ed.pts && ed.pts.length) ? ed.pts.length : 2) +
            '<br>Node indices: ' + ed.fromIdx + ' → ' + ed.toIdx,
          noDelete: true
        });
      });
    } else if (mode === 'marker') {
      (state.layoutMarkers || []).forEach(function(mk, idx) {
        if (!mk || seen['mk_' + mk.id]) return;
        seen['mk_' + mk.id] = true;
        if (mk.kind === 'text') {
          items.push({
            type: 'layoutMarker',
            id: mk.id,
            title: 'Marker | Text',
            tag: 'Text',
            details: escapeHtml(String(mk.text || '').slice(0, 200))
          });
        } else if (mk.kind === 'ruler') {
          const dx = Number(mk.x2) - Number(mk.x1), dy = Number(mk.y2) - Number(mk.y1);
          items.push({
            type: 'layoutMarker',
            id: mk.id,
            title: 'Marker | Ruler',
            tag: Math.hypot(dx, dy).toFixed(1) + ' m',
            details: 'Length: ' + Math.hypot(dx, dy).toFixed(1) + ' m'
          });
        } else if (mk.kind === 'island') {
          const nv = (mk.points && mk.points.length) || 0;
          items.push({
            type: 'layoutMarker',
            id: mk.id,
            title: 'Marker | Contour',
            tag: nv + ' vtx',
            details: 'Closed polygon · ' + nv + ' vertices (layout m = px)'
          });
        } else if (mk.kind === 'navaid') {
          const isIls = (mk.subType === 'ils');
          items.push({
            type: 'layoutMarker',
            id: mk.id,
            title: isIls ? 'Marker | ILS' : 'Marker | Lights',
            tag: isIls ? 'ILS' : '4',
            details: 'Navigation aid at (' + Number(mk.x).toFixed(1) + ', ' + Number(mk.y).toFixed(1) + ')'
          });
        } else if (mk.kind === 'area') {
          const nv = (mk.points && mk.points.length) || 0;
          items.push({
            type: 'layoutMarker',
            id: mk.id,
            title: 'Marker | Area',
            tag: nv + ' vtx',
            details: 'Filled polygon only · ' + nv + ' vertices · under other objects'
          });
        } else if (mk.kind === 'flight') {
          const tw = state.taxiways.find(function(t) { return t && t.id === mk.taxiwayId; });
          items.push({
            type: 'layoutMarker',
            id: mk.id,
            title: 'Marker | Flight',
            tag: 'Dummy',
            details: 'On: ' + (tw && tw.name ? tw.name : String(mk.taxiwayId))
          });
        }
      });
    }
    if (!items.length) {
      objectListEl.innerHTML = '<div class="obj-item">No objects yet.</div>';
      return;
    }
    objectListEl.innerHTML = items.map(it => (
      '<div class="obj-item" data-type="' + it.type + '" data-id="' + it.id + '">' +
        '<div class="obj-item-header">' +
          '<span class="obj-item-title">' + it.title + '</span>' +
          '<span class="obj-item-tag">' + it.tag + '</span>' +
          '<button type="button" class="obj-item-delete" title="Delete"' + (it.noDelete ? ' style="display:none" tabindex="-1" aria-hidden="true"' : '') + '>×</button>' +
        '</div>' +
        '<div class="obj-item-details">' + it.details + '</div>' +
      '</div>'
    )).join('');
    const listItems = objectListEl.querySelectorAll('.obj-item');
    listItems.forEach(el => {
      const type = el.getAttribute('data-type');
      const id = el.getAttribute('data-id');
      el.querySelector('.obj-item-delete').addEventListener('click', function(ev) {
        ev.stopPropagation();
        pushUndo();
        removeLayoutObjectFromState(type, id);
        if (state.selectedObject && state.selectedObject.type === type && state.selectedObject.id === id)
          state.selectedObject = null;
        if (type === 'terminal' && state.currentTerminalId === id) {
          state.currentTerminalId = state.terminals.length ? state.terminals[0].id : null;
          if (state.terminalDrawingId === id) {
            state.terminalDrawingId = null;
            state.layoutPathDrawPointer = null;
          }
        }
        if (type === 'taxiway' && state.taxiwayDrawingId === id) {
          state.taxiwayDrawingId = null;
          state.layoutPathDrawPointer = null;
        }
        syncPanelFromState();
        updateObjectInfo();
        if (typeof redrawLayoutAfterEdit === 'function') redrawLayoutAfterEdit();
        else if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths(); else draw();
      });
      el.addEventListener('click', function(ev) {
        if (ev.target.classList.contains('obj-item-delete')) return;
        const typ = this.getAttribute('data-type');
        const idr = this.getAttribute('data-id');
        if (typ === 'layoutEdge') rebuildDerivedGraphEdges();
        const obj = findLayoutObjectByListType(typ, idr);
        if (!obj) return;
        const wasExpanded = this.classList.contains('expanded');
        listItems.forEach(li => li.classList.remove('selected', 'expanded'));
        if (!wasExpanded) {
          this.classList.add('selected', 'expanded');
          state.flightPathRevealFlightId = null;
          state.selectedObject = { type: typ, id: idr, obj };
          if (typ === 'terminal') state.currentTerminalId = idr;
          if (typ === 'layoutMarker') {
            settingModeSelect.value = 'marker';
            if (typeof switchToTab === 'function') switchToTab('settings');
            syncMarkerSubKindTabFromSelectedLayoutMarker();
            if (typeof syncSettingsPaneToMode === 'function') syncSettingsPaneToMode();
          }
          focusCanvasForLayoutHotkeys();
          syncPanelFromState();
          updateObjectInfo();
        } else {
          objectInfoEl.className = 'object-info-panel is-empty';
          objectInfoEl.textContent = '';
        }
        draw();
      });
    });
    if (state.selectedObject) {
      const sel = objectListEl.querySelector('.obj-item[data-type="' + state.selectedObject.type + '"][data-id="' + state.selectedObject.id + '"]');
      if (sel) sel.classList.add('selected', 'expanded');
    }
  }

  /** Flight selection: compact schedule readout on the grid. Layout objects use #object-info in the floating left panel only (no duplicate HUD). */
  function updateFlightGridHud() {
    const el = document.getElementById('flight-grid-hud');
    if (!el) return;
    const sel = state.selectedObject;
    el.classList.remove('flight-grid-hud--layout-object');
    if (sel && sel.type === 'flight' && sel.obj) {
      const o = sel.obj;
      const fmtE = function(m) {
        return (typeof m === 'number' && isFinite(m)) ? formatMinutesToHHMMSS(m) : '—';
      };
      const ac = typeof getAircraftInfoByType === 'function' ? getAircraftInfoByType(o.aircraftType) : null;
      const typeLabel = ac ? (ac.name || ac.id || o.aircraftType || '—') : (o.aircraftType || '—');
      el.removeAttribute('hidden');
      el.innerHTML =
        '<div class="flight-grid-hud-reg">' + escapeHtml(o.reg || '—') + '</div>' +
        '<div class="flight-grid-hud-type">' + escapeHtml(typeLabel) + '</div>' +
        '<div class="flight-grid-hud-e">ELDT ' + escapeHtml(fmtE(o.eldtMin)) + ' · EIBT ' + escapeHtml(fmtE(o.eibtMin)) + '</div>' +
        '<div class="flight-grid-hud-e">EOBT ' + escapeHtml(fmtE(o.eobtMin)) + ' · ETOT ' + escapeHtml(fmtE(o.etotMin)) + '</div>';
    } else {
      el.setAttribute('hidden', '');
      el.innerHTML = '';
    }
  }

  function updatePathArcHud() {
    const root = document.getElementById('path-arc-hud');
    const btn = document.getElementById('btnPathArcToggle');
    const hint = document.getElementById('pathArcHudHint');
    if (!root || !btn) return;
    clearPathArcIfStale();
    btn.textContent = state.pathArcModeOn ? '호: ON' : '호: OFF';
    btn.setAttribute('aria-pressed', state.pathArcModeOn ? 'true' : 'false');
    btn.classList.toggle('path-arc-hud-toggle-on', !!state.pathArcModeOn);
    btn.title = state.pathArcModeOn
      ? 'OFF는 이 버튼만 누르세요. 호는 격자(캔버스)를 누른 채 드래그할 때만 그려집니다.'
      : 'ON 후 격자(캔버스)에서 누른 채 드래그하면 호가 만들어집니다.';
    const elig = isPathArcHudVertexSelection();
    const dragging = !!state.pathArcDrag;
    const show = !!(elig || dragging);
    if (!show) {
      root.setAttribute('hidden', '');
      root.style.display = 'none';
      if (hint) hint.setAttribute('hidden', '');
      return;
    }
    let vx, vy, pathSel;
    if (dragging) {
      const d = state.pathArcDrag;
      if (d.islandMarkerId != null) {
        const mkD = (state.layoutMarkers || []).find(function(m) { return m && String(m.id) === String(d.islandMarkerId); });
        const idxD = d.vertexIndex;
        if (!mkD || !isLayoutPolygonMarkerKind(mkD.kind) || !mkD.points || idxD < 0 || idxD >= mkD.points.length) {
          root.setAttribute('hidden', '');
          root.style.display = 'none';
          if (hint) hint.setAttribute('hidden', '');
          return;
        }
        const pv = mkD.points[idxD];
        vx = Number(pv.x);
        vy = Number(pv.y);
        pathSel = !!(state.selectedObject && state.selectedObject.type === 'layoutMarker' && String(state.selectedObject.id) === String(mkD.id));
      } else if (d.apronLinkId != null) {
        const lkD = (state.apronLinks || []).find(function(l) { return l && l.id === d.apronLinkId; });
        const polyD = lkD ? getApronLinkPolylineWorldPts(lkD) : [];
        const idxD = d.polyVertexIndex;
        if (!lkD || !polyD.length || idxD < 0 || idxD >= polyD.length) {
          root.setAttribute('hidden', '');
          root.style.display = 'none';
          if (hint) hint.setAttribute('hidden', '');
          return;
        }
        const pxy = polyD[idxD];
        vx = pxy[0];
        vy = pxy[1];
        pathSel = !!(state.selectedObject && state.selectedObject.type === 'apronLink' && state.selectedObject.id === lkD.id);
      } else {
        const twPos = state.taxiways.find(function(t) { return t.id === d.taxiwayId; });
        const idxPos = d.vertexIndex;
        if (!twPos || !twPos.vertices || idxPos < 0 || idxPos >= twPos.vertices.length) {
          root.setAttribute('hidden', '');
          root.style.display = 'none';
          if (hint) hint.setAttribute('hidden', '');
          return;
        }
        const v = twPos.vertices[idxPos];
        vx = cellToPixel(Number(v.col), Number(v.row))[0];
        vy = cellToPixel(Number(v.col), Number(v.row))[1];
        pathSel = !!(state.selectedObject && state.selectedObject.type === 'taxiway' && state.selectedObject.id === twPos.id);
      }
    } else {
      if (!elig) {
        root.setAttribute('hidden', '');
        root.style.display = 'none';
        if (hint) hint.setAttribute('hidden', '');
        return;
      }
      if (elig.kind === 'island') {
        const pv = elig.mk.points[elig.idx];
        vx = Number(pv.x);
        vy = Number(pv.y);
        pathSel = !!(state.selectedObject && state.selectedObject.type === 'layoutMarker' && String(state.selectedObject.id) === String(elig.mk.id));
      } else if (elig.kind === 'apronLink') {
        const polyE = getApronLinkPolylineWorldPts(elig.lk);
        const idxE = elig.polyVertexIndex;
        if (!polyE.length || idxE < 0 || idxE >= polyE.length) {
          root.setAttribute('hidden', '');
          root.style.display = 'none';
          if (hint) hint.setAttribute('hidden', '');
          return;
        }
        vx = polyE[idxE][0];
        vy = polyE[idxE][1];
        pathSel = !!(state.selectedObject && state.selectedObject.type === 'apronLink' && state.selectedObject.id === elig.lk.id);
      } else {
        const twPos = elig.tw;
        const idxPos = elig.idx;
        if (!twPos || !twPos.vertices || idxPos < 0 || idxPos >= twPos.vertices.length) {
          root.setAttribute('hidden', '');
          root.style.display = 'none';
          if (hint) hint.setAttribute('hidden', '');
          return;
        }
        const v = twPos.vertices[idxPos];
        vx = cellToPixel(Number(v.col), Number(v.row))[0];
        vy = cellToPixel(Number(v.col), Number(v.row))[1];
        pathSel = !!(state.selectedObject && state.selectedObject.type === 'taxiway' && state.selectedObject.id === twPos.id);
      }
    }
    const rWorld = layoutPathVertexRadiusPx(true, pathSel);
    const sc = worldToScreenCanvas(vx - rWorld, vy - rWorld);
    const left = Math.max(6, sc[0] - 8);
    const top = Math.max(6, sc[1] - 44);
    root.style.position = 'absolute';
    root.style.left = left.toFixed(1) + 'px';
    root.style.top = top.toFixed(1) + 'px';
    root.style.zIndex = '35';
    root.removeAttribute('hidden');
    root.style.display = 'flex';
    if (dragging) {
      if (hint) hint.removeAttribute('hidden');
    } else {
      if (hint) hint.setAttribute('hidden', '');
    }
  }

  function updateObjectInfo() {
    if (state.selectedObject) {
      objectInfoEl.className = 'object-info-panel has-selection';
      const o = state.selectedObject.obj;
      if (state.selectedObject.type === 'terminal') {
        const areaM2 = o.vertices && o.vertices.length >= 3 ? polygonAreaM2(o.vertices) : 0;
        const floors = o.floors != null ? Math.max(1, parseInt(o.floors, 10) || 1) : 1;
        const f2fRaw = o.floorToFloor != null ? Number(o.floorToFloor) : (o.floorHeight != null ? Number(o.floorHeight) : 4);
        const f2f = Math.max(0.5, f2fRaw || 4);
        const floorH = o.floorHeight != null ? Number(o.floorHeight) || (floors * f2f) : (floors * f2f);
        const totalArea = areaM2 * floors;
        const dep = o.departureCapacity != null ? o.departureCapacity : 0;
        const arr = o.arrivalCapacity != null ? o.arrivalCapacity : 0;
        objectInfoEl.innerHTML = '<strong>Building</strong><br>Name: ' + (o.name || o.id) + '<br>Type: ' + getBuildingTypeLabel(o.buildingType) + '<br>Vertices: ' + (o.vertices ? o.vertices.length : 0) +
          '<br>Footprint area: ' + areaM2.toFixed(1) + ' m²<br>Height: ' + floorH.toFixed(1) + ' m (Floors: ' + floors + ' × ' + f2f.toFixed(1) + ' m)' +
          '<br>Total floor area: ' + totalArea.toFixed(1) + ' m²' +
          '<br>Departure capacity: ' + dep + '<br>Arrival capacity: ' + arr;
      } else if (state.selectedObject.type === 'pbb') {
        const pConn = getStandConnectionPx(o);
        const pCell = [pConn[0] / CELL_SIZE, pConn[1] / CELL_SIZE];
        const pbbBuilding = getContactStandAttachedBuildingLabel(o);
        objectInfoEl.innerHTML =
          '<strong>Contact Stand</strong>' +
          '<br>Name: ' + (o.name || '—') +
          '<br>Constraint: ' + (getStandCategoryMode(o) === 'aircraft' ? 'Aircraft Type' : ('ICAO ' + (o.category || '—'))) +
          '<br>Category: ' + (o.category || '—') +
          '<br>Cell: (' + pCell[0].toFixed(1) + ',' + pCell[1].toFixed(1) + ')' +
          '<br>Angle: ' + getPbbAngleDeg(o).toFixed(0) + '°' +
          '<br>Building: ' + pbbBuilding +
          '<br>PBB count: ' + Math.max(1, parseInt(o.pbbCount, 10) || 1) +
          '<br>Edge cell: (' + o.edgeCol + ',' + o.edgeRow + ')';
      } else if (state.selectedObject.type === 'remote') {
        let allowedLabel = 'All (by proximity)';
        if (Array.isArray(o.allowedTerminals) && o.allowedTerminals.length) {
          const terms = makeUniqueNamedCopy(state.terminals || [], 'name').map(function(t) { return {
            id: t.id,
            name: (t.name || '').trim() || 'Building'
          }; });
          const names = o.allowedTerminals.map(function(id) {
            const tt = terms.find(function(t) { return t.id === id; });
            return tt ? tt.name : id;
          });
          if (names.length) allowedLabel = names.join(', ');
        }
        const remotePx = getRemoteStandCenterPx(o);
        const remoteCell = [remotePx[0] / CELL_SIZE, remotePx[1] / CELL_SIZE];
        objectInfoEl.innerHTML =
          '<strong>Remote stand</strong>' +
          '<br>Name: ' + (o.name || '—') +
          '<br>Constraint: ' + (getStandCategoryMode(o) === 'aircraft' ? 'Aircraft Type' : ('ICAO ' + (o.category || '—'))) +
          '<br>Cell: (' + remoteCell[0].toFixed(1) + ',' + remoteCell[1].toFixed(1) + ')' +
          '<br>available buildings: ' + allowedLabel;
      } else if (state.selectedObject.type === 'tempStand') {
        let allowedLabelT = 'All (by proximity)';
        if (Array.isArray(o.allowedTerminals) && o.allowedTerminals.length) {
          const terms = makeUniqueNamedCopy(state.terminals || [], 'name').map(function(t) { return {
            id: t.id,
            name: (t.name || '').trim() || 'Building'
          }; });
          const names = o.allowedTerminals.map(function(id) {
            const tt = terms.find(function(t) { return t.id === id; });
            return tt ? tt.name : id;
          });
          if (names.length) allowedLabelT = names.join(', ');
        }
        const tpx = getRemoteStandCenterPx(o);
        const tcell = [tpx[0] / CELL_SIZE, tpx[1] / CELL_SIZE];
        const junc = getTempStandTaxiwayJunctionPx(o);
        objectInfoEl.innerHTML =
          '<strong>Temp stand</strong>' +
          '<br>Name: ' + (o.name || '—') +
          '<br>Constraint: ' + (getStandCategoryMode(o) === 'aircraft' ? 'Aircraft Type' : ('ICAO ' + (o.category || '—'))) +
          '<br>Cell: (' + tcell[0].toFixed(1) + ',' + tcell[1].toFixed(1) + ')' +
          '<br>Taxiway junction (px): (' + junc[0].toFixed(1) + ', ' + junc[1].toFixed(1) + ') → sim_input junctionX/Y' +
          '<br>available buildings: ' + allowedLabelT +
          '<br>Placement: taxiway centerline (no apron link)';
      } else if (state.selectedObject.type === 'holdingPoint') {
        const hx = Number(o.x), hy = Number(o.y);
        const hCol = hx / CELL_SIZE, hRow = hy / CELL_SIZE;
        objectInfoEl.innerHTML =
          '<strong>' + holdingPointKindDisplayLabel(o.hpKind) + '</strong>' +
          '<br>Name: ' + (o.name || '—') +
          '<br>Diameter: ' + c2dHoldingPointDiameterM().toFixed(0) + ' m' +
          '<br>Cell: (' + hCol.toFixed(1) + ', ' + hRow.toFixed(1) + ')' +
          '<br>World: (' + hx.toFixed(0) + ', ' + hy.toFixed(0) + ')';
      }
      else if (state.selectedObject.type === 'taxiway') {
        const dirVal = getTaxiwayDirection(o);
        const dirLabel = dirVal === 'clockwise' ? 'Clockwise' : (dirVal === 'counter_clockwise' ? 'Counter Clockwise' : 'Both');
        const heading = o.pathType === 'runway' ? 'Runway' : (o.pathType === 'runway_exit' ? 'Runway Taxiway' : (o.pathType === 'apron_taxiway' ? 'Apron taxiway' : (o.pathType === 'general_queue_taxiway' ? 'Queue taxiway' : 'Taxiway')));
        const ser = serializeTaxiwayWithEndpoints(o);
        const startStr = ser.start_point != null ? '(' + ser.start_point.col + ', ' + ser.start_point.row + ')' : '—';
        const endStr = ser.end_point != null ? '(' + ser.end_point.col + ', ' + ser.end_point.row + ')' : '—';
        const avgVel = (typeof o.avgMoveVelocity === 'number' && isFinite(o.avgMoveVelocity) && o.avgMoveVelocity > 0) ? o.avgMoveVelocity : 10;
        const minArr = (o.pathType === 'runway')
          ? ((typeof o.minArrVelocity === 'number' && isFinite(o.minArrVelocity) && o.minArrVelocity > 0) ? Math.max(1, Math.min(150, o.minArrVelocity)) : 15)
          : null;
        const lineupStr = (o.pathType === 'runway')
          ? (String(getRunwayLineupDistMByDirection(o, 'clockwise')) + ' / ' +
             String(getRunwayLineupDistMByDirection(o, 'counter_clockwise')) +
             ' m (CW: from Start, CCW: from End)')
          : '';
        const maxEx = (o.pathType === 'runway_exit' && typeof o.maxExitVelocity === 'number' && isFinite(o.maxExitVelocity) && o.maxExitVelocity > 0) ? o.maxExitVelocity : null;
        const minEx = (o.pathType === 'runway_exit' && typeof o.minExitVelocity === 'number' && isFinite(o.minExitVelocity) && o.minExitVelocity > 0) ? o.minExitVelocity : null;
        const pavLabel = pathPavementResolvedForTaxiway(o) === 'cement' ? 'Cement' : 'Asphalt';
        objectInfoEl.innerHTML = '<strong>' + heading + '</strong><br>Name: ' + (o.name || '—') +
          '<br>Direction: ' + dirLabel +
          '<br>Width: ' + (o.width != null ? o.width : 23) + ' m' +
          '<br>Pavement: ' + pavLabel +
          ((o.pathType === 'taxiway' || o.pathType === 'apron_taxiway' || o.pathType === 'general_queue_taxiway') ? '<br>Avg move velocity: ' + avgVel + ' m/s' : '') +
          (minArr != null ? '<br>Min arr velocity: ' + minArr + ' m/s' : '') +
          (o.pathType === 'runway' ? '<br>Line up: ' + lineupStr : '') +
          (maxEx != null ? '<br>Max exit velocity: ' + maxEx + ' m/s' : '') +
          (minEx != null ? '<br>Min exit velocity: ' + minEx + ' m/s' : '') +
          '<br>Points: ' + (o.vertices ? o.vertices.length : 0) +
          '<br>Start point: ' + startStr + '<br>End point: ' + endStr;
      } else if (state.selectedObject.type === 'apronLink') {
        const lk = o;
        const stand = findStandById(lk.pbbId);
        const tw = state.taxiways.find(function(t) { return t.id === lk.taxiwayId; });
        objectInfoEl.innerHTML =
          '<strong>Apron Taxiway</strong><br>' +
          'Name: ' + getApronLinkDisplayName(lk) +
          '<br>Stand: ' + (stand && stand.name ? stand.name : lk.pbbId) +
          '<br>Taxiway: ' + (tw && tw.name ? tw.name : lk.taxiwayId) +
          '<br>Link point: (' + Number(lk.tx).toFixed(0) + ', ' + Number(lk.ty).toFixed(0) + ')';
      } else if (state.selectedObject.type === 'layoutMarker') {
        const mk = o;
        if (mk.kind === 'text') {
          const tid = 'layoutMarkerSelectedTextInput';
          objectInfoEl.innerHTML =
            '<strong>Marker · Text</strong><br>' +
            '<label for="' + tid + '">Text</label><br>' +
            '<input type="text" id="' + tid + '" class="object-info-text-input" style="width:100%;box-sizing:border-box;margin:6px 0;" value="' +
            escapeAttr(String(mk.text || '')) + '" spellcheck="false" />' +
            '<br>Position: (' + Number(mk.x).toFixed(1) + ', ' + Number(mk.y).toFixed(1) + ')';
          const tInp = document.getElementById(tid);
          if (tInp) {
            const markerId = mk.id;
            tInp.oninput = function() {
              const so = state.selectedObject;
              if (!so || so.type !== 'layoutMarker' || String(so.id) !== String(markerId)) return;
              const mo = so.obj;
              if (!mo || mo.kind !== 'text') return;
              mo.text = this.value;
              scheduleDraw();
            };
          }
        } else if (mk.kind === 'ruler') {
          const dx = Number(mk.x2) - Number(mk.x1), dy = Number(mk.y2) - Number(mk.y1);
          objectInfoEl.innerHTML = '<strong>Marker · Ruler</strong><br>Length: ' + Math.hypot(dx, dy).toFixed(1) + ' m' +
            '<br>From: (' + Number(mk.x1).toFixed(1) + ', ' + Number(mk.y1).toFixed(1) + ') → (' + Number(mk.x2).toFixed(1) + ', ' + Number(mk.y2).toFixed(1) + ')';
        } else if (mk.kind === 'island') {
          const nv = (mk.points && mk.points.length) || 0;
          const wid = 'layoutMarkerIslandWidthM';
          const wM = islandWidthMResolved(mk);
          objectInfoEl.innerHTML = '<strong>Marker · Contour</strong><br>Vertices: ' + nv +
            '<br><label for="' + wid + '">Width (m)</label><br>' +
            '<input type="number" id="' + wid + '" class="object-info-text-input" style="width:100%;box-sizing:border-box;margin:4px 0 8px;" min="0" max="' + LAYOUT_ISLAND_WIDTH_MAX_M + '" step="0.5" value="' + wM + '" inputmode="decimal" />' +
            '<br>호: 패널과 동일 토글 — 꼭짓점 선택 후 캔버스에서 드래그해 곡선 적용. Shift로 격자 스냅.' +
            '<br>닫기: 첫 점 근처 클릭(≥3점). 더블클릭으로 변 중간에 점 추가.';
          const markerId = mk.id;
          function bindIslandWidths() {
            const wEl = document.getElementById(wid);
            if (!wEl) return false;
            wEl.oninput = function() {
              const so = state.selectedObject;
              if (!so || so.type !== 'layoutMarker' || String(so.id) !== String(markerId)) return;
              const mo = so.obj;
              if (!mo || mo.kind !== 'island') return;
              const v = Number(this.value);
              mo.widthM = (isFinite(v) && v >= 0) ? Math.min(LAYOUT_ISLAND_WIDTH_MAX_M, v) : LAYOUT_ISLAND_WIDTH_DEFAULT_M;
              scheduleDraw();
              syncMarkerIslandSidebarWidthsFromSelection();
            };
            return true;
          }
          if (!bindIslandWidths()) setTimeout(bindIslandWidths, 0);
        } else if (mk.kind === 'area') {
          const nv = (mk.points && mk.points.length) || 0;
          objectInfoEl.innerHTML = '<strong>Marker · Area</strong><br>Vertices: ' + nv +
            '<br>Fill (when road width overlay is on) uses the same tint as other width bands; outline and editing work regardless.' +
            '<br>Close: click near first vertex (≥3 points). Double-click an edge to add a vertex. Path arc (호) works like Island.';
        } else if (mk.kind === 'flight') {
          const tw = state.taxiways.find(function(t) { return t && t.id === mk.taxiwayId; });
          const bid = 'layoutMarkerFlightBlazerToggle';
          ensureMarkerFlightBlazerState(mk);
          objectInfoEl.innerHTML = '<strong>Marker · Flight (dummy)</strong><br>Taxiway: ' + (tw && tw.name ? tw.name : String(mk.taxiwayId)) +
            '<br>Segment: ' + (mk.segIndex | 0) + ' · t: ' + Number(mk.t).toFixed(3) +
            '<br><button type="button" id="' + bid + '" class="small" style="margin-top:8px;display:inline-block;padding:6px 10px;border:1px solid var(--ui-border-default);border-radius:6px;background:var(--ui-bg-control);color:var(--ui-text-primary);cursor:pointer;">Blazer: ' + (mk.blazerEnabled ? 'ON' : 'OFF') + '</button>' +
            '<br>Trail points: L ' + mk.blazerLeftTrail.length + ' · R ' + mk.blazerRightTrail.length;
          const markerId = mk.id;
          function bindBlazerToggle() {
            const bEl = document.getElementById(bid);
            if (!bEl) return false;
            bEl.onclick = function() {
              const so = state.selectedObject;
              if (!so || so.type !== 'layoutMarker' || String(so.id) !== String(markerId) || !so.obj || so.obj.kind !== 'flight') return;
              ensureMarkerFlightBlazerState(so.obj);
              so.obj.blazerEnabled = !so.obj.blazerEnabled;
              if (so.obj.blazerEnabled) appendMarkerFlightBlazerTrail(so.obj);
              scheduleDraw();
              updateObjectInfo();
            };
            return true;
          }
          if (!bindBlazerToggle()) setTimeout(bindBlazerToggle, 0);
        } else if (mk.kind === 'navaid') {
          const sub = (mk.subType === 'ils') ? 'ils' : 'papi';
          const sid = 'layoutMarkerNavaidType';
          const heading = sub === 'ils' ? 'Marker · ILS' : 'Marker';
          objectInfoEl.innerHTML = '<strong>' + heading + '</strong>' +
            '<br><label for="' + sid + '">Nav aid type</label>' +
            '<br><select id="' + sid + '" class="object-info-text-input" style="width:100%;box-sizing:border-box;margin:6px 0;">' +
            '<option value="papi"' + (sub === 'papi' ? ' selected' : '') + '>PAPI</option>' +
            '<option value="ils"' + (sub === 'ils' ? ' selected' : '') + '>ILS</option>' +
            '</select>' +
            '<br>Position: (' + Number(mk.x).toFixed(1) + ', ' + Number(mk.y).toFixed(1) + ')';
          const markerId = mk.id;
          const sEl = document.getElementById(sid);
          if (sEl) sEl.onchange = function() {
            const so = state.selectedObject;
            if (!so || so.type !== 'layoutMarker' || String(so.id) !== String(markerId)) return;
            const mo = so.obj;
            if (!mo || mo.kind !== 'navaid') return;
            mo.subType = this.value === 'ils' ? 'ils' : 'papi';
            const panelSel = document.getElementById('markerNavaidType');
            if (panelSel) panelSel.value = mo.subType;
            scheduleDraw();
            updateObjectInfo();
            if (typeof renderObjectList === 'function') renderObjectList();
          };
        } else {
          objectInfoEl.innerHTML = '<strong>Marker</strong>';
        }
      } else if (state.selectedObject.type === 'layoutEdge') {
        const ed = state.selectedObject.obj;
        objectInfoEl.innerHTML =
          '<strong>Edge (derived)</strong><br>' +
          'Name: ' + getLayoutEdgeDisplayName(ed) +
          '<br>Graph length: ' + (ed && ed.dist != null ? Math.round(ed.dist) : '—') +
          '<br>Nodes: ' + (ed ? ed.fromIdx + ' → ' + ed.toIdx : '—') +
          '<br>Span (px): (' + (ed ? ed.x1.toFixed(0) : '—') + ', ' + (ed ? ed.y1.toFixed(0) : '—') + ') → (' + (ed ? ed.x2.toFixed(0) : '—') + ', ' + (ed ? ed.y2.toFixed(0) : '—') + ')' +
          '<br>Polyline points: ' + (ed && ed.pts ? ed.pts.length : 2);
      } else if (state.selectedObject.type === 'flight') {
        const dir = o.arrDep === 'Dep' ? 'Departure' : 'Arrival';
        const smInfo = flightScheduleMinutesForRow(o);
        const sibt = formatFlightScheduleDateTime(o, smInfo.sibt);
        const sobt = formatFlightScheduleDateTime(o, smInfo.sobt);
        const arrRunwayId = resolveArrivalRunwayIdForFlight(o);
        const arrRunwayObj = state.taxiways.find(function(tw) {
          return tw && tw.pathType === 'runway' && String(tw.id) === String(arrRunwayId || '');
        });
        const arrRwDir = normalizeRwDirectionValue(getTaxiwayDirection(arrRunwayObj));
        const landingDirLabel = arrRwDir === 'clockwise' ? 'CW' : (arrRwDir === 'counter_clockwise' ? 'CCW' : '—');
        const ac = typeof getAircraftInfoByType === 'function' ? getAircraftInfoByType(o.aircraftType) : null;
        const acName = ac ? (ac.name || ac.id || '') : (o.aircraftType || '—');
        const codeIcao = (ac && ac.icao) ? ac.icao : (o.code || '—');
        const icaoJhl = (ac && ac.icaoJHL) ? ac.icaoJHL : '—';
        const recatEu = (ac && ac.recatEu) ? ac.recatEu : '—';
        objectInfoEl.innerHTML =
          '<strong>Flight</strong><br>' +
          'Type: ' + dir +
          '<br>SIBT: ' + sibt + ' &nbsp; SOBT: ' + sobt +
          '<br>Aircraft: ' + (acName || '—') +
          '<br>Code(ICAO): ' + (codeIcao || '—') + ' &nbsp; ICAO(J/H/M/L): ' + (icaoJhl || '—') + ' &nbsp; RECAT-EU: ' + (recatEu || '—') +
          '<br>Landing direction: ' + landingDirLabel +
          '<br>Reg: ' + (o.reg || '—') +
          '<br>Airline Code: ' + (o.airlineCode || '—') + ' &nbsp; Flight Number: ' + (o.flightNumber || '—') +
          '<br>Dwell (Arr only): ' + (o.dwellMin || 0) + ' min';
      }
    } else {
      objectInfoEl.className = 'object-info-panel is-empty';
      objectInfoEl.textContent = '';
    }
    syncMarkerIslandSidebarWidthsFromSelection();
    updateFlightGridHud();
    updatePathArcHud();
    renderObjectList();
  }

  function reset2DView() {
    let w = 0, h = 0;
    const rect = container.getBoundingClientRect();
    w = Number(rect.width) || 0;
    h = Number(rect.height) || 0;
    if (w <= 0 || h <= 0) {
      if (canvas) {
        w = canvas.clientWidth || canvas.width || 800;
        h = canvas.clientHeight || canvas.height || 600;
      } else {
        w = 800;
        h = 600;
      }
    }
    w = Math.max(1, w);
    h = Math.max(1, h);
    const maxX = GRID_COLS * CELL_SIZE;
    const maxY = GRID_ROWS * CELL_SIZE;
    const scaleX = w / maxX;
    const scaleY = h / maxY;
    const s = Math.min(scaleX, scaleY) * 0.9;
    state.scale = s;
    state.panX = (w - maxX * s) / 2;
    state.panY = (h - maxY * s) / 2;
    draw();
  }

  function resizeCanvas() {
    if (!container || !canvas || !ctx) return;
    const rect = container.getBoundingClientRect();
    const w = Math.max(1, Number(rect.width) || 0);
    const h = Math.max(1, Number(rect.height) || 0);
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
    if (overlayCanvas && overlayCtx) {
      overlayCanvas.width = w * dpr;
      overlayCanvas.height = h * dpr;
      overlayCanvas.style.width = w + 'px';
      overlayCanvas.style.height = h + 'px';
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    invalidateGridUnderlay();
    syncLayoutHeatmapSvgViewBox();
    safeDraw({ bypassSimScrubGuard: true });
    syncMarkerTextDraftInputPosition();
  }

  let _gridUnderlayCanvas = null;
  let _gridUnderlayDirty = true;
  function invalidateGridUnderlay() { _gridUnderlayDirty = true; }
  function rebuildGridUnderlay() {
    const maxX = GRID_COLS * CELL_SIZE, maxY = GRID_ROWS * CELL_SIZE;
    if (!_gridUnderlayCanvas) _gridUnderlayCanvas = document.createElement('canvas');
    _gridUnderlayCanvas.width = Math.max(1, Math.floor(maxX * dpr));
    _gridUnderlayCanvas.height = Math.max(1, Math.floor(maxY * dpr));
    const uctx = _gridUnderlayCanvas.getContext('2d');
    uctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    uctx.fillStyle = GRID_VIEW_BG;
    uctx.fillRect(0, 0, maxX, maxY);
    if (state.layoutImageOverlay && layoutImageBitmap) {
      const overlay = state.layoutImageOverlay;
      const [imgX, imgY] = cellToPixel(overlay.topLeftCol, overlay.topLeftRow);
      uctx.save();
      uctx.globalAlpha = state.layers.image ? clampLayoutImageOpacity(overlay.opacity) : 0;
      uctx.imageSmoothingEnabled = true;
      uctx.drawImage(
        layoutImageBitmap,
        imgX,
        imgY,
        clampLayoutImageSize(overlay.widthM, GRID_LAYOUT_IMAGE_DEFAULTS.widthM),
        clampLayoutImageSize(overlay.heightM, GRID_LAYOUT_IMAGE_DEFAULTS.heightM)
      );
      uctx.restore();
    }
    _gridUnderlayDirty = false;
  }

  function drawGrid(interactiveLite) {
    const w = layoutDrawCanvas.width / dpr, h = layoutDrawCanvas.height / dpr;
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = GRID_VIEW_BG;
    ctx.fillRect(0, 0, w, h);
    ctx.restore();
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.translate(state.panX, state.panY);
    ctx.scale(state.scale, state.scale);
    const maxX = GRID_COLS * CELL_SIZE, maxY = GRID_ROWS * CELL_SIZE;
    if (_gridUnderlayDirty) rebuildGridUnderlay();
    ctx.drawImage(_gridUnderlayCanvas, 0, 0, maxX, maxY);
    if (!state.layers.grid) {
      ctx.restore();
      return;
    }
    const drawMinor = !(GRID_MINOR_GRID_MIN_SCALE > 0 && state.scale < GRID_MINOR_GRID_MIN_SCALE);
    const marginWorld = GRID_DRAW_VIEWPORT_MARGIN_CELLS * CELL_SIZE;
    const s = state.scale || 1;
    const minWx = (0 - state.panX) / s - marginWorld;
    const maxWx = (w - state.panX) / s + marginWorld;
    const minWy = (0 - state.panY) / s - marginWorld;
    const maxWy = (h - state.panY) / s + marginWorld;
    const cMin = Math.max(0, Math.floor(minWx / CELL_SIZE));
    const cMax = Math.min(GRID_COLS, Math.ceil(maxWx / CELL_SIZE));
    const rMin = Math.max(0, Math.floor(minWy / CELL_SIZE));
    const rMax = Math.min(GRID_ROWS, Math.ceil(maxWy / CELL_SIZE));
    for (let c = cMin; c <= cMax; c++) {
      const isMajor = (c % GRID_MAJOR_INTERVAL === 0);
      if (!isMajor && !drawMinor) continue;
      const x = c * CELL_SIZE;
      ctx.strokeStyle = isMajor
        ? ('rgba(' + GRID_MAJOR_LINE_RGB + ',' + GRID_MAJOR_LINE_OPACITY + ')')
        : ('rgba(' + GRID_MINOR_LINE_RGB + ',' + GRID_MINOR_LINE_OPACITY + ')');
      ctx.lineWidth = isMajor ? GRID_MAJOR_LINE_WIDTH : GRID_MINOR_LINE_WIDTH;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, maxY);
      ctx.stroke();
    }
    for (let r = rMin; r <= rMax; r++) {
      const isMajor = (r % GRID_MAJOR_INTERVAL === 0);
      if (!isMajor && !drawMinor) continue;
      const y = r * CELL_SIZE;
      ctx.strokeStyle = isMajor
        ? ('rgba(' + GRID_MAJOR_LINE_RGB + ',' + GRID_MAJOR_LINE_OPACITY + ')')
        : ('rgba(' + GRID_MINOR_LINE_RGB + ',' + GRID_MINOR_LINE_OPACITY + ')');
      ctx.lineWidth = isMajor ? GRID_MAJOR_LINE_WIDTH : GRID_MINOR_LINE_WIDTH;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(maxX, y);
      ctx.stroke();
    }
    if (!interactiveLite) {
      ctx.fillStyle = '#aaa';
      ctx.font = '10px system-ui';
      ctx.fillText('0,0', 4, 2);
    }
    const cx = (GRID_COLS * CELL_SIZE) / 2;
    const cy = (GRID_ROWS * CELL_SIZE) / 2;
    ctx.beginPath();
    ctx.fillStyle = '#ef4444';
    ctx.arc(cx, cy, CELL_SIZE * 0.15, 0, Math.PI * 2);
    ctx.fill();
    if (state.hoverCell != null) {
      const hc = state.hoverCell;
      const hx = hc.col * CELL_SIZE;
      const hy = hc.row * CELL_SIZE;
      ctx.beginPath();
      ctx.fillStyle = 'rgba(248, 113, 113, 0.45)';
      ctx.arc(hx, hy, CELL_SIZE * 0.2, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }

  function layoutHairlineStrokeWidthWorld() {
    return 1 / Math.max(state.scale || 1, 0.02);
  }
  const LAYOUT_AREA_DIAGONAL_HATCH_SPACING_M = 10;
  const LAYOUT_PBB_BRIDGE_HALF_WIDTH_M = 1;
  function hatchIntersectionsLineXMinusYEqualsC(c, rx0, rx1, ry0, ry1) {
    const pts = [];
    function add(x, y) {
      if (x < rx0 - 1e-9 || x > rx1 + 1e-9 || y < ry0 - 1e-9 || y > ry1 + 1e-9) return;
      for (let i = 0; i < pts.length; i++) {
        if (Math.hypot(pts[i][0] - x, pts[i][1] - y) < 1e-6) return;
      }
      pts.push([x, y]);
    }
    add(rx0, rx0 - c);
    add(rx1, rx1 - c);
    add(ry0 + c, ry0);
    add(ry1 + c, ry1);
    if (pts.length < 2) return null;
    pts.sort(function(a, b) { return a[0] !== b[0] ? a[0] - b[0] : a[1] - b[1]; });
    return [pts[0][0], pts[0][1], pts[pts.length - 1][0], pts[pts.length - 1][1]];
  }
  function drawPolygonDiagonalHatch45M(hatchCtx, points, spacingM, strokeStyle, lineWidthWorld, strokeGlobalAlpha) {
    if (!hatchCtx || !Array.isArray(points) || points.length < 3 || !isFinite(spacingM) || spacingM <= 0) return;
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    points.forEach(function(p) {
      minX = Math.min(minX, p[0]);
      maxX = Math.max(maxX, p[0]);
      minY = Math.min(minY, p[1]);
      maxY = Math.max(maxY, p[1]);
    });
    const span = Math.max(maxX - minX, maxY - minY, spacingM);
    const pad = span * 2 + spacingM * 4;
    const rx0 = minX - pad, rx1 = maxX + pad, ry0 = minY - pad, ry1 = maxY + pad;
    const cCornerMin = Math.min(rx0 - ry0, rx1 - ry0, rx0 - ry1, rx1 - ry1);
    const cCornerMax = Math.max(rx0 - ry0, rx1 - ry0, rx0 - ry1, rx1 - ry1);
    const step = spacingM * Math.SQRT2;
    hatchCtx.save();
    if (typeof strokeGlobalAlpha === 'number' && isFinite(strokeGlobalAlpha)) {
      const ga = Math.max(0, Math.min(1, strokeGlobalAlpha));
      hatchCtx.globalAlpha = (typeof hatchCtx.globalAlpha === 'number' ? hatchCtx.globalAlpha : 1) * ga;
    }
    hatchCtx.beginPath();
    hatchCtx.moveTo(points[0][0], points[0][1]);
    for (let i = 1; i < points.length; i++) hatchCtx.lineTo(points[i][0], points[i][1]);
    hatchCtx.closePath();
    hatchCtx.clip();
    hatchCtx.strokeStyle = strokeStyle;
    hatchCtx.lineWidth = lineWidthWorld;
    hatchCtx.setLineDash([]);
    for (let k = Math.floor(cCornerMin / step) * step; k <= cCornerMax + step * 0.5; k += step) {
      const seg = hatchIntersectionsLineXMinusYEqualsC(k, rx0, rx1, ry0, ry1);
      if (!seg) continue;
      hatchCtx.beginPath();
      hatchCtx.moveTo(seg[0], seg[1]);
      hatchCtx.lineTo(seg[2], seg[3]);
      hatchCtx.stroke();
    }
    hatchCtx.restore();
  }
  function drawPolygonHatch(points, strokeStyle, spacingPx) {
    if (!Array.isArray(points) || points.length < 3) return;
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    points.forEach(function(p) {
      minX = Math.min(minX, p[0]);
      maxX = Math.max(maxX, p[0]);
      minY = Math.min(minY, p[1]);
      maxY = Math.max(maxY, p[1]);
    });
    const span = Math.max(maxX - minX, maxY - minY);
    const pad = span + Math.max(40, spacingPx * 2);
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(points[0][0], points[0][1]);
    for (let i = 1; i < points.length; i++) ctx.lineTo(points[i][0], points[i][1]);
    ctx.closePath();
    ctx.clip();
    ctx.strokeStyle = strokeStyle;
    ctx.lineWidth = layoutHairlineStrokeWidthWorld();
    ctx.setLineDash([]);
    for (let offset = minX - pad; offset <= maxX + pad; offset += spacingPx) {
      ctx.beginPath();
      ctx.moveTo(offset, maxY + pad);
      ctx.lineTo(offset + (maxY - minY) + pad, minY - pad);
      ctx.stroke();
    }
    ctx.restore();
  }
  function drawTerminals(interactiveLite) {
    const vb = layoutWorldViewportAabbWithBufferM(LAYOUT_RENDER_VIEWPORT_BUFFER_M);
    const nowPerf = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
    const suppressBuildingFill = !!state.isPanning || nowPerf < _layoutDetailSuppressUntil;
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.translate(state.panX, state.panY);
    ctx.scale(state.scale, state.scale);
    state.terminals.forEach(term => {
      const isDrawingTerm = state.terminalDrawingId === term.id;
      if (term.vertices.length === 0 && !isDrawingTerm) return;
      if (!isDrawingTerm) {
        const termAabb = terminalWorldAabbFromVertices(term);
        if (termAabb && !aabbIntersectsViewport(vb, termAabb)) return;
      }
      const selected = state.selectedObject && state.selectedObject.type === 'terminal' && state.selectedObject.id === term.id;
      const buildingTheme = getBuildingTheme(term);
      const termPts = term.vertices.map(function(v) { return cellToPixel(v.col, v.row); });
      const ptrTerm = isDrawingTerm ? state.layoutPathDrawPointer : null;
      const hoverTerm = (ptrTerm && ptrTerm.length >= 2) ? ptrTerm : null;
      if (isDrawingTerm && !term.closed && term.vertices.length >= 1) {
        strokeLayoutPathDraftPolyline(ctx, termPts, hoverTerm);
        drawLayoutPathDraftVertexDots(ctx, termPts, hoverTerm);
      } else {
        const hairW = layoutHairlineStrokeWidthWorld();
        const bl = !!state.layers.buildingLines;
        const bf = !!state.layers.buildingFill && !suppressBuildingFill;
        const useFillMono = layerMonoFillOn() && !selected && bf;
        const useLineMono = layerMonoLinesOn() && !selected;
        const monoFillB = c2dLayerMonoFillDarkAsphaltCss();
        const monoLineB = c2dLayerMonoLineStrokeCss();
        if (bl || bf || selected) {
          ctx.lineWidth = hairW;
          ctx.strokeStyle = selected ? c2dObjectSelectedStroke() : (useLineMono ? monoLineB : buildingTheme.stroke);
          ctx.fillStyle = selected ? c2dObjectSelectedFill() : (useFillMono ? monoFillB : buildingTheme.fill);
          ctx.beginPath();
          for (let i = 0; i < termPts.length; i++) {
            const [x,y] = termPts[i];
            if (i === 0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
          }
          if (term.closed) {
            ctx.closePath();
            if (buildingTheme.fillEnabled && bf) ctx.fill();
          }
          ctx.shadowBlur = 0;
          if (term.closed && buildingTheme.fillEnabled && bf) {
            drawPolygonDiagonalHatch45M(
              ctx,
              termPts,
              LAYOUT_AREA_DIAGONAL_HATCH_SPACING_M,
              selected ? c2dObjectSelectedStroke() : (useLineMono ? monoLineB : buildingTheme.stroke),
              hairW,
              0.8
            );
          }
          if (bl) {
            ctx.beginPath();
            for (let i = 0; i < termPts.length; i++) {
              const [x,y] = termPts[i];
              if (i === 0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
            }
            if (term.closed) ctx.closePath();
            ctx.stroke();
            if (!interactiveLite && term.closed && term.vertices.length > 0 && normalizeBuildingType(term.buildingType) !== 'hanger') {
              let cx = 0, cy = 0;
              term.vertices.forEach(v => {
                const [px, py] = cellToPixel(v.col, v.row);
                cx += px; cy += py;
              });
              cx /= term.vertices.length;
              cy /= term.vertices.length;
              const label = term.name || term.id || 'Building';
              ctx.fillStyle = useLineMono ? monoLineB : buildingTheme.labelFill;
              ctx.font = '12px system-ui';
              ctx.textAlign = 'center';
              ctx.textBaseline = 'middle';
              ctx.fillText(label, cx, cy);
            }
          }
        }
      }
      if (selected && !(isDrawingTerm && !term.closed && term.vertices.length >= 1)) {
        term.vertices.forEach((v, i) => {
          const [x,y] = cellToPixel(v.col, v.row);
          const vertexSelected = isSelectedVertex('terminal', term.id, i);
          ctx.beginPath();
          ctx.fillStyle = vertexSelected ? '#f43f5e' : (i === 0 ? '#f97316' : '#e5e7eb');
          ctx.arc(x, y, layoutTerminalVertexRadiusPx(vertexSelected), 0, Math.PI*2);
          ctx.fill();
        });
      }
    });
    ctx.restore();
  }

  function drawPBBs(interactiveLite) {
    const vb = layoutWorldViewportAabbWithBufferM(LAYOUT_RENDER_VIEWPORT_BUFFER_M);
    const nowPerf = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
    const suppressStandFill = !!state.isPanning || nowPerf < _layoutDetailSuppressUntil;
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.translate(state.panX, state.panY);
    ctx.scale(state.scale, state.scale);
    state.pbbStands.forEach(pbb => {
      const x1 = Number(pbb.x1), y1 = Number(pbb.y1), x2 = Number(pbb.x2), y2 = Number(pbb.y2);
      if (!Number.isFinite(x1) || !Number.isFinite(y1) || !Number.isFinite(x2) || !Number.isFinite(y2)) return;
      const pbbSelForCull = state.selectedObject && state.selectedObject.type === 'pbb' && state.selectedObject.id === pbb.id;
      if (!pbbSelForCull) {
        const pbbAabb = pointsWorldAabb(getPBBStandCorners(pbb));
        if (pbbAabb && !aabbIntersectsViewport(vb, pbbAabb)) return;
      }
      rebuildPbbBridgeGeometry(pbb);
      const depP = getStandDepthMeters(pbb.category || 'C');
      const widP = getStandWidthMeters(pbb.category || 'C');
      const sel = state.selectedObject && state.selectedObject.type === 'pbb' && state.selectedObject.id === pbb.id;
      const simOcc = state.hasSimulationResult && isStandOccupiedAtSimSec(pbb.id, state.simTimeSec);
      const sl = !!state.layers.standLines, sf = !!state.layers.standFill && !suppressStandFill;
      const monoFillP = layerMonoFillOn() && !sel;
      const monoLineP = layerMonoLinesOn() && !sel;
      const monoFp = c2dLayerMonoFillDarkAsphaltCss();
      const monoLp = c2dLayerMonoLineStrokeCss();
      if (!sl && !sf && !sel) return;
      if (!interactiveLite) drawPbbBoardingRectangle(ctx, pbb, sel);
      const bridges = Array.isArray(pbb.pbbBridges) ? pbb.pbbBridges : [];
      bridges.forEach(function(bridge, bridgeIdx) {
        const pts = Array.isArray(bridge.points) ? bridge.points : [];
        if (pts.length < 3) return;
        const hairW = layoutHairlineStrokeWidthWorld();
        const bx = Number(pts[1].x) || 0, by = Number(pts[1].y) || 0;
        const px = Number(pts[2].x) || 0, py = Number(pts[2].y) || 0;
        const dx = px - bx, dy = py - by;
        const L = Math.hypot(dx, dy) || 1e-6;
        const tx = dx / L, ty = dy / L;
        const nx = -ty, ny = tx;
        const hw = LAYOUT_PBB_BRIDGE_HALF_WIDTH_M;
        const quad = [
          [bx + nx * hw, by + ny * hw],
          [bx - nx * hw, by - ny * hw],
          [px - nx * hw, py - ny * hw],
          [px + nx * hw, py + ny * hw],
        ];
        ctx.shadowBlur = 0;
        ctx.beginPath();
        ctx.moveTo(quad[0][0], quad[0][1]);
        for (let qi = 1; qi < 4; qi++) ctx.lineTo(quad[qi][0], quad[qi][1]);
        ctx.closePath();
        if (sf && !interactiveLite) {
          if (monoFillP) {
            ctx.fillStyle = c2dLayerMonoFillDarkAsphaltRgba(0.52);
            ctx.fill();
            drawPolygonDiagonalHatch45M(ctx, quad, LAYOUT_AREA_DIAGONAL_HATCH_SPACING_M, monoLp, hairW);
          } else {
            ctx.fillStyle = sel ? 'rgba(255,255,255,0.16)' : 'rgba(255,255,255,0.1)';
            ctx.fill();
            drawPolygonDiagonalHatch45M(ctx, quad, LAYOUT_AREA_DIAGONAL_HATCH_SPACING_M, sel ? 'rgba(255,255,255,0.45)' : 'rgba(255,255,255,0.38)', hairW);
          }
        }
        if (sl && !interactiveLite) {
          ctx.beginPath();
          ctx.moveTo(quad[0][0], quad[0][1]);
          for (let qi = 1; qi < 4; qi++) ctx.lineTo(quad[qi][0], quad[qi][1]);
          ctx.closePath();
          ctx.strokeStyle = monoLineP ? monoLp : 'rgba(255,255,255,0.9)';
          ctx.lineWidth = hairW;
          ctx.stroke();
        }
        if (sel) {
          [0, 2].forEach(function(ptIdx) {
            const pt = pts[ptIdx];
            if (!pt) return;
            const isBridgeVertexSelected = !!(state.selectedVertex && state.selectedVertex.type === 'pbbBridge' && state.selectedVertex.id === pbb.id && state.selectedVertex.bridgeIndex === bridgeIdx && state.selectedVertex.pointIndex === ptIdx);
            ctx.beginPath();
            ctx.fillStyle = isBridgeVertexSelected ? '#f43f5e' : '#e5e7eb';
            ctx.arc(Number(pt.x) || 0, Number(pt.y) || 0, isBridgeVertexSelected ? 4 : 3, 0, Math.PI * 2);
            ctx.fill();
          });
        }
      });
      const apronPt = getStandConnectionPx(pbb);
      const ex = apronPt[0], ey = apronPt[1];
      const angle = getPBBStandAngle(pbb);
      const rotationActive = !!(state.selectedVertex && state.selectedVertex.type === 'standRotation' && state.selectedVertex.id === pbb.id);
      const apronLinked = standHasApronTaxiwayLink(pbb.id);
      const idleFill = monoFillP
        ? c2dLayerMonoFillDarkAsphaltRgba(apronLinked ? 0.4 : 0.48)
        : (apronLinked ? 'rgba(14,92,40,0.26)' : 'rgba(52,56,64,0.42)');
      ctx.save();
      ctx.translate(ex, ey);
      ctx.rotate(angle);
      ctx.setLineDash([]);
      if (sf && !interactiveLite) {
        ctx.fillStyle = sel
          ? c2dObjectSelectedFill()
          : (simOcc ? (monoFillP ? c2dLayerMonoFillDarkAsphaltRgba(0.58) : c2dSimStandOccupiedFill()) : idleFill);
        fillStandSafetyFootprintInLocalAxes(ctx, depP, widP, pbb.category || 'C');
      }
      if (sl) {
        drawStandSafetyContourInLocalAxes(ctx, depP, widP, pbb.category || 'C', sel);
        if (!interactiveLite) drawStandApronMarkingsInLocalAxes(ctx, depP, widP, pbb.category || 'C');
        if (!interactiveLite) {
          const nameRaw = (pbb.name && pbb.name.trim()) ? pbb.name.trim() : String(state.pbbStands.indexOf(pbb) + 1);
          const label = nameRaw;
          const pad = 3;
          const tx = depP / 2 - pad;
          const ty = -widP / 2 + pad;
          ctx.fillStyle = monoLineP ? monoLp : (apronLinked ? '#dcd8cf' : '#d1d5db');
          ctx.font = '8px system-ui';
          ctx.textAlign = 'right';
          ctx.textBaseline = 'top';
          ctx.fillText(String(label), tx, ty);
        }
      }
      ctx.restore();
      if (sl || sel) {
        const acMk = getStandAircraftMarkerWorldPxForPbb(pbb);
        const fill = sel ? c2dObjectSelectedStroke() : (monoLineP ? monoLp : (apronLinked ? c2dTaxiwayCenterlineStroke() : 'rgba(156,163,175,0.95)'));
        drawApronSiteMarker(ctx, acMk[0], acMk[1], fill, null, sel, angle);
      }
      if (sel) {
        drawStandRotationHandle(getPbbRotationOriginPx(pbb), getPbbRotationHandlePx(pbb), rotationActive);
      }
    });
    ctx.restore();
  }

  function drawRemoteStands(interactiveLite) {
    const vb = layoutWorldViewportAabbWithBufferM(LAYOUT_RENDER_VIEWPORT_BUFFER_M);
    const nowPerf = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
    const suppressStandFill = !!state.isPanning || nowPerf < _layoutDetailSuppressUntil;
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.translate(state.panX, state.panY);
    ctx.scale(state.scale, state.scale);
    state.remoteStands.forEach(st => {
      const remoteSelForCull = state.selectedObject && state.selectedObject.type === 'remote' && state.selectedObject.id === st.id;
      if (!remoteSelForCull) {
        const remoteAabb = pointsWorldAabb(getRemoteStandCorners(st));
        if (remoteAabb && !aabbIntersectsViewport(vb, remoteAabb)) return;
      }
      const [cx, cy] = getRemoteStandCenterPx(st);
      const depR = getStandDepthMeters(st.category || 'C');
      const widR = getStandWidthMeters(st.category || 'C');
      const angle = getRemoteStandAngleRad(st);
      const sel = state.selectedObject && state.selectedObject.type === 'remote' && state.selectedObject.id === st.id;
      const simOcc = state.hasSimulationResult && isStandOccupiedAtSimSec(st.id, state.simTimeSec);
      const rotationActive = !!(state.selectedVertex && state.selectedVertex.type === 'standRotation' && state.selectedVertex.id === st.id);
      const apronLinkedR = standHasApronTaxiwayLink(st.id);
      const monoFillR = layerMonoFillOn() && !sel;
      const monoLineR = layerMonoLinesOn() && !sel;
      const monoLr = c2dLayerMonoLineStrokeCss();
      const idleFillR = monoFillR
        ? c2dLayerMonoFillDarkAsphaltRgba(apronLinkedR ? 0.4 : 0.48)
        : (apronLinkedR ? 'rgba(14,92,40,0.26)' : 'rgba(52,56,64,0.42)');
      const sl = !!state.layers.standLines, sf = !!state.layers.standFill && !suppressStandFill;
      if (!sl && !sf && !sel) return;
      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(angle);
      ctx.setLineDash([]);
      if (sf && !interactiveLite) {
        ctx.fillStyle = sel
          ? c2dObjectSelectedFill()
          : (simOcc ? (monoFillR ? c2dLayerMonoFillDarkAsphaltRgba(0.58) : c2dSimStandOccupiedFill()) : idleFillR);
        fillStandSafetyFootprintInLocalAxes(ctx, depR, widR, st.category || 'C');
      }
      if (sl) {
        drawStandSafetyContourInLocalAxes(ctx, depR, widR, st.category || 'C', sel);
        if (!interactiveLite) drawStandApronMarkingsInLocalAxes(ctx, depR, widR, st.category || 'C');
        if (!interactiveLite) {
          const nameRaw = (st.name && st.name.trim()) ? st.name.trim() : ('R' + String(state.remoteStands.indexOf(st) + 1).padStart(3, '0'));
          const label = nameRaw;
          const pad = 3;
          const tx = depR / 2 - pad;
          const ty = -widR / 2 + pad;
          ctx.fillStyle = monoLineR ? monoLr : (apronLinkedR ? '#dcd8cf' : '#d1d5db');
          ctx.font = '8px system-ui';
          ctx.textAlign = 'right';
          ctx.textBaseline = 'top';
          ctx.fillText(String(label), tx, ty);
        }
      }
      ctx.restore();
      if (sl || sel) {
        const rm = getStandAircraftMarkerWorldPxForRemoteLike(st);
        const fill = sel ? c2dObjectSelectedStroke() : (monoLineR ? monoLr : (apronLinkedR ? c2dTaxiwayCenterlineStroke() : 'rgba(156,163,175,0.95)'));
        drawApronSiteMarker(ctx, rm[0], rm[1], fill, null, sel, angle);
      }
      if (sel) {
        drawStandRotationHandle([cx, cy], getRemoteRotationHandlePx(st), rotationActive);
      }
    });
    ctx.restore();
  }

  function drawTempStands(interactiveLite) {
    const vb = layoutWorldViewportAabbWithBufferM(LAYOUT_RENDER_VIEWPORT_BUFFER_M);
    const nowPerf = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
    const suppressStandFill = !!state.isPanning || nowPerf < _layoutDetailSuppressUntil;
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.translate(state.panX, state.panY);
    ctx.scale(state.scale, state.scale);
    const mode = settingModeSelect ? settingModeSelect.value : 'grid';
    const temps = state.tempStands || [];
    temps.forEach(function(st) {
      const tempSelForCull = state.selectedObject && state.selectedObject.type === 'tempStand' && state.selectedObject.id === st.id;
      if (!tempSelForCull) {
        const tempAabb = pointsWorldAabb(getRemoteStandCorners(st));
        if (tempAabb && !aabbIntersectsViewport(vb, tempAabb)) return;
      }
      const cxcy = getRemoteStandCenterPx(st);
      const cx = cxcy[0], cy = cxcy[1];
      const depT = getStandDepthMeters(st.category || 'C');
      const widT = getStandWidthMeters(st.category || 'C');
      const angle = getRemoteStandAngleRad(st);
      const sel = state.selectedObject && state.selectedObject.type === 'tempStand' && state.selectedObject.id === st.id;
      const simOcc = state.hasSimulationResult && isStandOccupiedAtSimSec(st.id, state.simTimeSec);
      const rotationActive = !!(state.selectedVertex && state.selectedVertex.type === 'standRotation' && state.selectedVertex.id === st.id);
      const monoFillT = layerMonoFillOn() && !sel;
      const monoLineT = layerMonoLinesOn() && !sel;
      const monoLt = c2dLayerMonoLineStrokeCss();
      const idleFillT = monoFillT ? c2dLayerMonoFillDarkAsphaltRgba(0.5) : 'rgba(139,92,246,0.2)';
      const idx = temps.indexOf(st);
      const nameRaw = (st.name && st.name.trim()) ? st.name.trim() : ('T' + String(idx + 1).padStart(3, '0'));
      const labelPrefix = getStandCategoryMode(st) === 'aircraft' ? 'AC' : (st.category || 'C');
      const label = labelPrefix + ' / ' + nameRaw;
      const sl = !!state.layers.standLines, sf = !!state.layers.standFill && !suppressStandFill;
      if (!sl && !sf && !sel) return;
      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(angle);
      ctx.setLineDash([]);
      if (sf && !interactiveLite) {
        ctx.fillStyle = sel
          ? c2dObjectSelectedFill()
          : (simOcc ? (monoFillT ? c2dLayerMonoFillDarkAsphaltRgba(0.58) : c2dSimStandOccupiedFill()) : idleFillT);
        fillStandSafetyFootprintInLocalAxes(ctx, depT, widT, st.category || 'C');
      }
      if (sl) {
        drawStandSafetyContourInLocalAxes(ctx, depT, widT, st.category || 'C', sel);
        if (!interactiveLite) drawStandApronMarkingsInLocalAxes(ctx, depT, widT, st.category || 'C');
        if (!interactiveLite) {
          ctx.setLineDash([]);
          const pad = 3;
          const tx = depT / 2 - pad;
          const ty = -widT / 2 + pad;
          ctx.fillStyle = monoLineT ? monoLt : '#e9d5ff';
          ctx.font = '8px system-ui';
          ctx.textAlign = 'right';
          ctx.textBaseline = 'top';
          ctx.fillText(String(label), tx, ty);
        }
      }
      ctx.restore();
      if (mode === 'apronTaxiway' && (sl || sel)) {
        const tm = getStandAircraftMarkerWorldPxForRemoteLike(st);
        const apronLinkedT = standHasApronTaxiwayLink(st.id);
        const fill = sel ? '#c4b5fd' : (monoLineT ? monoLt : (apronLinkedT ? c2dTaxiwayCenterlineStroke() : '#7c3aed'));
        drawApronSiteMarker(ctx, tm[0], tm[1], fill, null, sel, angle);
      }
      const junc = getTempStandTaxiwayJunctionPx(st);
      const jx = junc[0], jy = junc[1];
      const jr = Math.max(3.2, 3.6 / Math.max(state.scale, 0.08));
      if (!!state.layers.junction && (sl || sel)) {
        ctx.save();
        ctx.setLineDash([]);
        const etcM = layerMonoEtcOn() && !sel;
        ctx.strokeStyle = etcM ? C2D_LAYER_MONO_ETC_WHITE : (sel ? '#22d3ee' : 'rgba(34,211,238,0.95)');
        ctx.lineWidth = sel ? 2.25 : 1.75;
        ctx.beginPath();
        ctx.arc(jx, jy, jr, 0, Math.PI * 2);
        ctx.stroke();
        ctx.fillStyle = etcM ? C2D_LAYER_MONO_ETC_WHITE : (sel ? '#ecfeff' : 'rgba(236,254,255,0.92)');
        ctx.beginPath();
        ctx.arc(jx, jy, jr * 0.45, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      }
      if (sel) {
        drawStandRotationHandle([cx, cy], getRemoteRotationHandlePx(st), rotationActive);
      }
    });
    ctx.restore();
  }

  function renderRunwaySeparation() {
    const panel = document.getElementById('rwySepPanel');
    if (!panel) return;
    const runways = (state.taxiways || []).filter(tw => tw.pathType === 'runway');
    if (!runways.length) {
      panel.innerHTML = '<div style="font-size:11px;color:#9ca3af;">No runway paths. Layout Mode <strong>Runway</strong>Draw the runway polyline first with.</div>';
      return;
    }
    if (!state.activeRwySepId || !runways.some(r => r.id === state.activeRwySepId)) {
      state.activeRwySepId = runways[0].id;
    }
    const active = runways.find(r => r.id === state.activeRwySepId) || runways[0];
    const cfg = rsepGetConfigForRunway(active);
    const stdKey = cfg.standard || 'ICAO';
    const cats = RSEP_STD_CATS[stdKey] || RSEP_STD_CATS['ICAO'];
    const mode = cfg.mode || 'MIX';
    const seq = cfg.activeSeq || (RSEP_MODE_SEQS[mode] && RSEP_MODE_SEQS[mode][0]) || 'ARR→ARR';
    const seqType = RSEP_SEQ_TYPES[seq] || 'matrix';
    const seqMeta = rsepGetSeqMeta(seq);

    let html = '';
    html += '<div class="rwysep-rwy-bar">';
    html += '<div class="rwysep-rwy-tabs">';
    runways.forEach(rw => {
      const isActive = rw.id === active.id;
      const label = escapeHtml(rw.name || ('Runway ' + rw.id));
      html += '<button type="button" class="rwysep-rwy-btn' + (isActive ? ' active' : '') + '" data-rwy-id="' + rw.id + '">' + label + '</button>';
    });
    html += '</div></div>';

    const activeSub = 'noname';
    html += '<div class="layout-save-load-tabs" style="margin-top:8px;">';
    html += '<button type="button" class="layout-save-load-tab rwysep-subtab-btn active" data-subtab="noname">No Name</button>';
    html += '</div>';

    html += '<div id="rwysep-subtab-input" style="">';
    html += '<div class="rwysep-block">';
    html += '<div class="rwysep-label">Standard &amp; Mode</div>';
    html += '<div class="rwysep-row">';
    html += '<label style="font-size:11px;color:#9ca3af;">Standard&nbsp;</label>';
    html += '<select id="rwysep-standard">';
    html += '<option value="ICAO"' + (stdKey === 'ICAO' ? ' selected' : '') + '>ICAO (J/H/M/L)</option>';
    html += '<option value="RECAT-EU"' + (stdKey === 'RECAT-EU' ? ' selected' : '') + '>RECAT-EU (A~F)</option>';
    html += '</select>';
    html += '<label style="font-size:11px;color:#9ca3af;margin-left:8px;">Mode&nbsp;</label>';
    html += '<select id="rwysep-mode">';
    ['ARR','DEP','MIX'].forEach(m => {
      const txt = m === 'ARR' ? 'Arrivals only' : (m === 'DEP' ? 'Departures only' : 'Mixed (Arr/Dep)');
      html += '<option value="' + m + '"' + (mode === m ? ' selected' : '') + '>' + txt + '</option>';
    });
    html += '</select>';
    html += '<label style="font-size:11px;color:#9ca3af;margin-left:8px;">Seq&nbsp;</label>';
    html += '<select id="rwysep-seq">';
    (RSEP_MODE_SEQS[mode] || []).forEach(s => {
      const lbl = s;
      html += '<option value="' + s + '"' + (seq === s ? ' selected' : '') + '>' + lbl + '</option>';
    });
    html += '</select>';
    html += '</div></div>';

    if (seqMeta) {
      html += '<div class="rwysep-block" style="margin-top:4px;">';
      html += '<div class="rwysep-label">Concept summary</div>';
      html += '<div style="font-size:10px;color:#d1d5db;line-height:1.5;background:#020617;border-radius:6px;border:1px solid #111827;padding:6px 8px;">';
      html += '<div><span style="color:#9ca3af;">Driving factor</span>&nbsp;&nbsp;: ' + escapeHtml(seqMeta.driver) + '</div>';
      html += '<div><span style="color:#9ca3af;">Reference point</span>&nbsp;: ' + escapeHtml(seqMeta.refPoint) + '</div>';
      html += '<div><span style="color:#9ca3af;">Input structure</span>: ' + escapeHtml(seqMeta.input) + '</div>';
      html += '</div>';
      html += '</div>';
    }

    if (seq === 'ARR→DEP') {
      html += '<div class="rwysep-block">';
      html += '<div style="font-size:10px;color:#9ca3af;line-height:1.5;margin-bottom:6px;">Separation combines leading aircraft ROT with trailing aircraft lineup–gear-off time, using the ROT inputs above per wake category.</div>';

      const totalRot = cats.length;
      let filledRot = 0;
      cats.forEach(c => {
        const val = cfg.rot && cfg.rot[c] != null ? cfg.rot[c] : '';
        if (val !== '' && val != null) filledRot += 1;
      });
      html += rsepLegendHtml(filledRot, totalRot);

      html += '<div class="rwysep-row" style="flex-wrap:wrap;">';
      cats.forEach(c => {
        const rawVal = cfg.rot && cfg.rot[c] != null ? cfg.rot[c] : '';
        const valStr = rawVal === null || rawVal === undefined ? '' : String(rawVal);
        const sub = rsepGetCatLabel(stdKey, c);
        const colInfo = rsepColorForValue(valStr);
        html += '<div style="min-width:90px;margin-right:6px;margin-bottom:4px;">';
        html += '<div style="font-size:10px;color:#9ca3af;margin-bottom:2px;line-height:1.2;">';
        html += 'Cat ' + c;
        if (sub) {
          html += '<div style="font-size:9px;color:#6b7280;margin-top:1px;">' + escapeHtml(sub) + '</div>';
        }
        html += '</div>';
        html += '<input type="number" min="0" step="5" data-rwysep-rot="' + c + '" value="' + escapeHtml(valStr) + '" style="width:64px;background:' + colInfo.bg + ';border:1px solid ' + colInfo.border + ';color:' + colInfo.color + ';font-size:10px;padding:3px 4px;border-radius:3px;text-align:center;" />';
        html += ' <span style="font-size:9px;color:#6b7280;">sec</span>';
        html += '</div>';
      });
      html += '</div></div>';
    }

    if (seq !== 'ARR→DEP') {
      html += '<div class="rwysep-block">';
      html += '<div class="rwysep-label">Separation (sec) — ' + escapeHtml(seq) + '</div>';
      if (seqType === 'matrix') {
        const data = cfg.seqData && cfg.seqData[seq] ? cfg.seqData[seq] : rsepMakeMatrix(cats, null);
        const total = cats.length * cats.length;
        let filled = 0;
        cats.forEach(lead => {
          cats.forEach(trail => {
            const v = data[lead] && data[lead][trail] != null ? data[lead][trail] : '';
            if (v !== '' && v != null) filled += 1;
          });
        });
        html += rsepLegendHtml(filled, total);

        html += '<div class="rwysep-matrix-wrap"><table class="rwysep-table"><thead><tr>';
        html += '<th>Lead↓ / Trail→</th>';
        cats.forEach(c => {
          const sub = rsepGetCatLabel(stdKey, c);
          html += '<th><div style="line-height:1.2;">' + c;
          if (sub) {
            html += '<div style="font-size:9px;color:#9ca3af;margin-top:1px;">' + escapeHtml(sub) + '</div>';
          }
          html += '</div></th>';
        });
        html += '</tr></thead><tbody>';
        cats.forEach(lead => {
          const leadSub = rsepGetCatLabel(stdKey, lead);
          html += '<tr><td><div style="line-height:1.2;">' + lead;
          if (leadSub) {
            html += '<div style="font-size:9px;color:#9ca3af;margin-top:1px;">' + escapeHtml(leadSub) + '</div>';
          }
          html += '</div></td>';
          cats.forEach(trail => {
            const v = data[lead] && data[lead][trail] != null ? data[lead][trail] : '';
            const colInfo = rsepColorForValue(v);
            html += '<td><input type="number" min="0" step="5" data-rwysep-matrix-lead="' + lead + '" data-rwysep-matrix-trail="' + trail + '" value="' + escapeHtml(String(v)) + '" style="width:64px;background:' + colInfo.bg + ';border:1px solid ' + colInfo.border + ';color:' + colInfo.color + ';font-size:10px;padding:3px 4px;border-radius:3px;text-align:center;" /></td>';
          });
          html += '</tr>';
        });
        html += '</tbody></table></div>';
      } else {
        const data1d = cfg.seqData && cfg.seqData[seq] ? cfg.seqData[seq] : rsepMake1D(cats, null);
        const total = cats.length;
        let filled = 0;
        cats.forEach(cat => {
          const v = data1d[cat] != null ? data1d[cat] : '';
          if (v !== '' && v != null) filled += 1;
        });
        html += rsepLegendHtml(filled, total);

        html += '<div class="rwysep-row" style="flex-wrap:wrap;margin-top:4px;">';
        cats.forEach(cat => {
          const v = data1d[cat] != null ? data1d[cat] : '';
          const colInfo = rsepColorForValue(v);
          const sub = rsepGetCatLabel(stdKey, cat);
          html += '<div style="min-width:90px;margin-right:6px;margin-bottom:4px;border:1px solid #1f2937;border-radius:6px;padding:6px 8px;background:#020617;">';
          html += '<div style="font-size:10px;color:#9ca3af;margin-bottom:2px;line-height:1.2;">Cat ' + cat;
          if (sub) {
            html += '<div style="font-size:9px;color:#6b7280;margin-top:1px;">' + escapeHtml(sub) + '</div>';
          }
          html += '</div>';
          html += '<input type="number" min="0" step="5" data-rwysep-1d="' + cat + '" value="' + escapeHtml(String(v)) + '" style="width:64px;background:' + colInfo.bg + ';border:1px solid ' + colInfo.border + ';color:' + colInfo.color + ';font-size:10px;padding:3px 4px;border-radius:3px;text-align:center;" />';
          html += ' <span style="font-size:9px;color:#6b7280;">sec</span>';
          html += '</div>';
        });
        html += '</div>';
      }
      html += '</div>';
    }
    html += '</div>'; // end subtab input

    html += '<div id="rwysep-subtab-timeline" style="' + (activeSub === 'timeline' ? '' : 'display:none;') + '">';
    html += '<div class="rwysep-block" style="margin-top:8px;">';
    html += '<div class="rwysep-label">Separation Timeline (Reg × Time)</div>';
    html += '<div id="rwySepTimeWrap" style="width:100%;background:#020617;border-radius:8px;border:1px solid #1f2937;position:relative;overflow-x:auto;overflow-y:auto;margin-top:4px;max-height:calc(40px * 12 + 80px);"></div>';
    html += '<div style="font-size:9px;color:#9ca3af;margin-top:4px;">';
    html += 'Y: Reg Number · X: Time · Bars = S-series (SLDT–STOT) · Lines = E-series (ELDT–ETOT)';
    html += '</div></div>';
    html += '</div>'; // end subtab timeline

    panel.innerHTML = html;

    function drawRwySeparationTimeline() {
      if (state.activeRwySepSubtab && state.activeRwySepSubtab !== 'timeline') return;
      const wrap = panel.querySelector('#rwySepTimeWrap');
      if (!wrap) return;

      const allData = typeof buildRunwaySeparationTimelineByRunwaySnapshot === 'function'
        ? buildRunwaySeparationTimelineByRunwaySnapshot(state.flights)
        : null;
      const data = allData && active && active.id != null ? allData[active.id] : null;
      if (!data || !data.events || !data.events.length) {
        wrap.innerHTML = '<div style="font-size:11px;color:#9ca3af;padding:8px 10px;">No SLDT/STOT events for this runway.</div>';
        return;
      }

      const byFlight = new Map();
      data.events.forEach(ev => {
        const f = ev.flight;
        if (!f) return;
        let lane = byFlight.get(f);
        if (!lane) {
          const reg = f.reg || f.id || '';
          lane = {
            flight: f,
            reg,
            hasArr: false,
            hasDep: false,
            sldt: null,
            eldt: null,
            stot: null,
            etot: null
          };
          byFlight.set(f, lane);
        }
        if (ev.type === 'arr') {
          lane.hasArr = true;
          lane.sldt = ev.time;
          lane.eldt = (f.eldtMin != null ? f.eldtMin : ev.time);
        } else if (ev.type === 'dep') {
          lane.hasDep = true;
          lane.stot = ev.time;
          lane.etot = (f.etotMin != null ? f.etotMin : ev.time);
        }
      });

      const lanes = Array.from(byFlight.values());
      if (!lanes.length) {
        wrap.innerHTML = '<div style="font-size:11px;color:#9ca3af;padding:8px 10px;">No SLDT/STOT events for this runway.</div>';
        return;
      }

      let minT0 = Infinity;
      let maxT0 = -Infinity;
      lanes.forEach(ln => {
        if (ln.sldt != null && ln.sldt < minT0) minT0 = ln.sldt;
        if (ln.etot != null && ln.etot > maxT0) maxT0 = ln.etot;
      });
      if (minT0 <= 0 && lanes.length) {
        const pos = lanes.map(function(ln) { return ln.sldt; }).filter(function(v) { return v != null && isFinite(v) && v > 1e-6; });
        if (pos.length) minT0 = Math.min.apply(null, pos);
      }
      if (!isFinite(minT0) || !isFinite(maxT0)) {
        minT0 = data.minT;
        maxT0 = data.maxT;
      }
      let baseMinT = Math.max(0, minT0 - RWY_SEP_TIMELINE_PAD_MIN);
      let baseMaxT = maxT0 + RWY_SEP_TIMELINE_PAD_MIN;
      if (baseMaxT <= baseMinT) baseMaxT = baseMinT + 60;
      const baseSpan = baseMaxT - baseMinT;
      const zoom = (state.rwySepTimeZoom && state.rwySepTimeZoom > 1) ? state.rwySepTimeZoom : 1;
      const span = baseSpan;
      const minT = baseMinT;
      const maxT = baseMaxT;

      lanes.sort((a, b) => {
        const ta = (a.sldt ?? a.stot ?? a.eldt ?? a.etot ?? 0);
        const tb = (b.sldt ?? b.stot ?? b.eldt ?? b.etot ?? 0);
        return ta - tb;
      });

      const tickPositions = buildTimeAxisTicks(minT, maxT, baseMinT, baseSpan, zoom);

      const sMarkers = [];
      const eMarkers = [];

      const rows = [];
      lanes.forEach(ln => {
        const reg = ln.reg || '';
        const sStart = (ln.sldt != null ? ln.sldt : null);
        const sEnd = (ln.stot != null ? ln.stot : null);
        const eStart = (ln.eldt != null ? ln.eldt : null);
        const eEnd = (ln.etot != null ? ln.etot : null);

        let blocks = '';
        if (sStart != null && sEnd != null && span > 0) {
          const s1 = Math.max(sStart, baseMinT);
          const s2 = Math.min(sEnd, baseMaxT);
          if (s2 <= s1) return;
          const leftPct = ((s1 - baseMinT) / baseSpan) * 100 * zoom;
          const widthPct = Math.max(1, ((s2 - s1) / baseSpan) * 100 * zoom);
          const rightPct = leftPct + widthPct;
          sMarkers.push({ t: sStart, leftPct, type: 'start' });
          sMarkers.push({ t: sEnd, leftPct: rightPct, type: 'end' });
          blocks +=
            '<div class="rwysep-line-s" style="' +
              'left:' + leftPct + '%;' +
              'width:' + widthPct + '%;' +
            '"></div>' +
            '<div class="rwysep-tri" style="' +
              'top:20%;' +
              'left:' + leftPct + '%;' +
              'border-top:6px solid ' + GANTT_COLORS.S_SERIES + ';' +
            '"></div>' +
            '<div class="rwysep-tri" style="' +
              'top:20%;' +
              'left:' + rightPct + '%;' +
              'border-bottom:6px solid ' + GANTT_COLORS.S_SERIES + ';' +
            '"></div>';
        }
        if (eStart != null && eEnd != null && span > 0) {
          const e1 = Math.max(eStart, baseMinT);
          const e2 = Math.min(eEnd, baseMaxT);
          if (e2 <= e1) return;
          const leftPct2 = ((e1 - baseMinT) / baseSpan) * 100 * zoom;
          const widthPct2 = Math.max(0.5, ((e2 - e1) / baseSpan) * 100 * zoom);
          const rightPct2 = leftPct2 + widthPct2;
          eMarkers.push({ t: eStart, leftPct: leftPct2, type: 'start' });
          eMarkers.push({ t: eEnd, leftPct: rightPct2, type: 'end' });
          blocks +=
            '<div class="rwysep-line-e" style="' +
              'left:' + leftPct2 + '%;' +
              'width:' + widthPct2 + '%;' +
            '"></div>' +
            '<div class="rwysep-tri" style="' +
              'top:54%;' +
              'left:' + leftPct2 + '%;' +
              'border-top:6px solid ' + GANTT_COLORS.E_SERIES + ';' +
            '"></div>' +
            '<div class="rwysep-tri" style="' +
              'top:54%;' +
              'left:' + rightPct2 + '%;' +
              'border-bottom:6px solid ' + GANTT_COLORS.E_SERIES + ';' +
            '"></div>';
        }

        rows.push(
          '<div class="alloc-row">' +
            '<div class="alloc-row-label">' + escapeHtml(reg) + '</div>' +
            '<div class="alloc-row-track" style="background:transparent;border:none;">' + blocks + '</div>' +
          '</div>'
        );
      });

      sMarkers.sort((a, b) => a.t - b.t);
      eMarkers.sort((a, b) => a.t - b.t);

      const sHeadMarks = sMarkers.map(m =>
        '<div class="rwysep-tri" style="' +
          'top:60%;' +
          'left:' + m.leftPct + '%;' +
          (m.type === 'start'
            ? 'border-top:6px solid ' + GANTT_COLORS.S_SERIES + ';'
            : 'border-bottom:6px solid ' + GANTT_COLORS.S_SERIES + ';') +
        '"></div>'
      ).join('');

      const eHeadMarks = eMarkers.map(m =>
        '<div class="rwysep-tri" style="' +
          'top:60%;' +
          'left:' + m.leftPct + '%;' +
          (m.type === 'start'
            ? 'border-top:6px solid ' + GANTT_COLORS.E_SERIES + ';'
            : 'border-bottom:6px solid ' + GANTT_COLORS.E_SERIES + ';') +
        '"></div>'
      ).join('');

      const headHtml =
        '<div class="rwysep-head-row">' +
          '<div class="rwysep-head-label">S-series</div>' +
          '<div class="rwysep-head-track">' + sHeadMarks + '</div>' +
        '</div>' +
        '<div class="rwysep-head-row">' +
          '<div class="rwysep-head-label">E-series</div>' +
          '<div class="rwysep-head-track">' + eHeadMarks + '</div>' +
        '</div>';

      const axisTicks = tickPositions.map(tp =>
        '<div class="alloc-time-tick" style="left:' + tp.leftPct + '%;">' +
          '<div class="alloc-time-tick-label">' + tp.label + '</div>' +
        '</div>'
      );
      const axisHtml =
        '<div class="alloc-time-axis-overlay">' +
          '<div class="alloc-time-axis-inner">' + axisTicks.join('') + '</div>' +
        '</div>';

      const rwyGridOverlay =
        '<div class="alloc-gantt-grid-overlay">' +
          tickPositions.map(function(tp) {
            return '<div class="alloc-time-grid-line" style="left:' + tp.leftPct + '%;"></div>';
          }).join('') +
        '</div>';
      const rowsHtml = '<div class="rwysep-rows">' + rwyGridOverlay + rows.join('') + '</div>';
      wrap.innerHTML = headHtml + rowsHtml + axisHtml;

      if (!wrap._rwySepZoomBound) {
        wrap._rwySepZoomBound = true;
        wrap.addEventListener('wheel', function(e) {
          if (!e.shiftKey) return;
          e.preventDefault();
          const factor = e.deltaY < 0 ? 1.15 : (1 / 1.15);
          let z = state.rwySepTimeZoom || 1;
          z *= factor;
          if (z < 1) z = 1;
          if (z > 8) z = 8;
          state.rwySepTimeZoom = z;
          if (typeof renderRunwaySeparation === 'function') renderRunwaySeparation();
        }, { passive: false });
      }

      if (!wrap._rwySepScrollBound) {
        wrap._rwySepScrollBound = true;
        wrap.addEventListener('scroll', function() {
          if (wrap._rwySepScrollRecalc) return;
          const currentLeft = wrap.scrollLeft;
          wrap._rwySepScrollRecalc = true;
          drawRwySeparationTimeline();
          wrap.scrollLeft = currentLeft;
          wrap._rwySepScrollRecalc = false;
        });
      }
    }

    drawRwySeparationTimeline();

    _rwySepWireInputHandlers(panel, cfg, cats, seq, state);
  }

  function _rwySepWireInputHandlers(panel, cfg, cats, seq, st) {
    panel.querySelectorAll('.rwysep-rwy-btn').forEach(function(btn) {
      btn.addEventListener('click', function() {
        const id = this.getAttribute('data-rwy-id');
        if (!id) return;
        st.activeRwySepId = id;
        renderRunwaySeparation();
      });
    });
    panel.querySelectorAll('.rwysep-subtab-btn').forEach(function(btn) {
      btn.addEventListener('click', function() {
        const sub = this.getAttribute('data-subtab') || 'input';
        st.activeRwySepSubtab = sub;
        renderRunwaySeparation();
      });
    });
    var stdSel = panel.querySelector('#rwysep-standard');
    if (stdSel) {
      stdSel.addEventListener('change', function() {
        cfg.standard = this.value || 'ICAO';
        cfg.seqData = rsepMakeSeqData(cfg.standard);
        var catsNew = RSEP_STD_CATS[cfg.standard] || [];
        var rotNew = RSEP_STANDARDS[cfg.standard] && RSEP_STANDARDS[cfg.standard].ROT || {};
        cfg.rot = {};
        catsNew.forEach(function(c) { cfg.rot[c] = rotNew[c] != null ? String(rotNew[c]) : ''; });
        renderRunwaySeparation();
      });
    }
    var modeSel = panel.querySelector('#rwysep-mode');
    if (modeSel) {
      modeSel.addEventListener('change', function() {
        cfg.mode = this.value || 'MIX';
        var seqs = RSEP_MODE_SEQS[cfg.mode] || ['ARR→ARR'];
        if (!seqs.includes(cfg.activeSeq)) cfg.activeSeq = seqs[0];
        renderRunwaySeparation();
      });
    }
    var seqSel = panel.querySelector('#rwysep-seq');
    if (seqSel) {
      seqSel.addEventListener('change', function() {
        cfg.activeSeq = this.value || 'ARR→ARR';
        renderRunwaySeparation();
      });
    }
    function _applyColorOnChange(inp) {
      var colInfo = rsepColorForValue(inp.value);
      inp.style.background = colInfo.bg;
      inp.style.borderColor = colInfo.border;
      inp.style.color = colInfo.color;
    }
    panel.querySelectorAll('input[data-rwysep-rot]').forEach(function(inp) {
      inp.addEventListener('change', function() {
        var cat = this.getAttribute('data-rwysep-rot');
        if (!cat) return;
        cfg.rot[cat] = this.value;
        _applyColorOnChange(this);
      });
    });
    panel.querySelectorAll('input[data-rwysep-matrix-lead]').forEach(function(inp) {
      inp.addEventListener('change', function() {
        var lead = this.getAttribute('data-rwysep-matrix-lead');
        var trail = this.getAttribute('data-rwysep-matrix-trail');
        if (!lead || !trail) return;
        if (!cfg.seqData[seq]) cfg.seqData[seq] = rsepMakeMatrix(cats, null);
        if (!cfg.seqData[seq][lead]) cfg.seqData[seq][lead] = {};
        cfg.seqData[seq][lead][trail] = this.value;
        _applyColorOnChange(this);
      });
    });
    panel.querySelectorAll('input[data-rwysep-1d]').forEach(function(inp) {
      inp.addEventListener('change', function() {
        var cat = this.getAttribute('data-rwysep-1d');
        if (!cat) return;
        if (!cfg.seqData[seq]) cfg.seqData[seq] = rsepMake1D(cats, null);
        cfg.seqData[seq][cat] = this.value;
        _applyColorOnChange(this);
      });
    });
  }

  function drawTaxiways(interactiveLite) {
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.translate(state.panX, state.panY);
    ctx.scale(state.scale, state.scale);
    const _twVb = layoutWorldViewportAabbWithBufferM(LAYOUT_RENDER_VIEWPORT_BUFFER_M);
    const _twPanLite = !!(state.isPanning && layoutViewportWidthWorldM() > LAYOUT_DRAG_LITE_VIEWPORT_WIDTH_THRESHOLD_M);
    const _twPanCoarse = !!(_twPanLite && (state.scale || 1) < 0.4);
    function centerlineWidthWorld(baseWorld, minScreenPx) {
      const s = Math.max(0.08, state.scale || 1);
      return Math.max(baseWorld, minScreenPx / s);
    }
    function strokeTwCenterlineYellow(yellowStroke) {
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';
      ctx.lineWidth = centerlineWidthWorld(1, 0.8);
      ctx.strokeStyle = yellowStroke;
      ctx.stroke();
    }
    function taxiwayDrawContext(tw) {
      const drawing = state.taxiwayDrawingId === tw.id;
      if (tw.vertices.length < 2 && !drawing) return null;
      const isRunwayPath = tw.pathType === 'runway';
      const isRunwayExit = tw.pathType === 'runway_exit';
      const isApronTaxiwayPath = tw.pathType === 'apron_taxiway';
      const widthDefault = isRunwayPath ? RUNWAY_PATH_DEFAULT_WIDTH : (isRunwayExit ? RUNWAY_EXIT_DEFAULT_WIDTH : TAXIWAY_DEFAULT_WIDTH);
      const width = tw.width != null ? tw.width : widthDefault;
      const sel = state.selectedObject && state.selectedObject.type === 'taxiway' && state.selectedObject.id === tw.id;
      const pathLineCap = 'butt';
      const pathFillWide = !!state.layers.pathFill && !state.isPanning;
      return { drawing, isRunwayPath, isRunwayExit, isApronTaxiwayPath, width, widthDefault, sel, pathLineCap, pathFillWide };
    }
    function forEachTaxiwayInPavementUnderlayOrder(callback) {
      if (!state.layers.pathFill) {
        state.taxiways.forEach(callback);
        return;
      }
      state.taxiways.forEach(function(tw) {
        if (pathPavementResolvedForTaxiway(tw) === 'cement') callback(tw);
      });
      state.taxiways.forEach(function(tw) {
        if (pathPavementResolvedForTaxiway(tw) === 'asphalt') callback(tw);
      });
    }
    forEachTaxiwayInPavementUnderlayOrder(tw => {
      if (!taxiwayShouldDrawInViewport(tw, _twVb)) return;
      const g = taxiwayDrawContext(tw);
      if (!g) return;
      if (interactiveLite && g.isApronTaxiwayPath && !state.isPanning) return;
      const drawing = g.drawing, isRunwayPath = g.isRunwayPath, isRunwayExit = g.isRunwayExit, isApronTaxiwayPath = g.isApronTaxiwayPath, width = g.width, sel = g.sel, pathLineCap = g.pathLineCap, pathFillWide = g.pathFillWide;
      let strokeC, fillC;
      if (sel) {
        strokeC = c2dObjectSelectedStroke();
        fillC = c2dObjectSelectedFill();
      } else if (isRunwayPath || isRunwayExit) {
        strokeC = c2dRunwayStroke();
        fillC = c2dRunwayFill();
      } else if (isApronTaxiwayPath) {
        strokeC = drawing ? 'rgba(123, 121, 109, 0.88)' : c2dTaxiwayPavementStroke();
        fillC = drawing ? 'rgba(123, 121, 109, 0.16)' : c2dTaxiwayPavementFill();
      } else {
        strokeC = drawing ? 'rgba(74, 74, 74, 0.82)' : c2dRunwayStroke();
        fillC = drawing ? 'rgba(74, 74, 74, 0.16)' : c2dRunwayFill();
      }
      if (pathFillWide) {
        if (!sel) {
          const bandCss = c2dRoadWidthBandForPavement(pathPavementResolvedForTaxiway(tw));
          strokeC = bandCss;
          fillC = bandCss;
        } else {
          strokeC = c2dCssColorRgbChannelScale(c2dCssColorToOpaque(strokeC), ROAD_WIDTH_SURFACE_RGB_MUL);
          fillC = c2dCssColorRgbChannelScale(c2dCssColorToOpaque(fillC), ROAD_WIDTH_SURFACE_RGB_MUL);
        }
      }
      if (pathFillWide && layerMonoFillOn() && !sel) {
        const m = c2dLayerMonoFillDarkAsphaltCss();
        strokeC = m;
        fillC = m;
      }
      ctx.strokeStyle = strokeC;
      ctx.fillStyle = fillC;
      if (pathFillWide) {
        ctx.lineWidth = width;
        ctx.lineCap = pathLineCap;
        ctx.lineJoin = 'round';
        ctx.beginPath();
        for (let i = 0; i < tw.vertices.length; i++) {
          const [x, y] = cellToPixel(tw.vertices[i].col, tw.vertices[i].row);
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        if (tw.vertices.length >= 2) {
          if (sel) {
            ctx.save();
            ctx.shadowColor = c2dObjectSelectedGlow();
            ctx.shadowBlur = c2dObjectSelectedGlowBlur();
            ctx.stroke();
            ctx.restore();
          } else ctx.stroke();
        }
      }
      if (!interactiveLite && (!_twPanCoarse || sel || drawing) && isRunwayPath && tw.vertices.length >= 2 && pathFillWide) {
        const runwayPts = tw.vertices.map(v => cellToPixel(v.col, v.row));
        drawRunwayDecorations(tw, runwayPts, width, { baseOnly: true });
      }
    });
    state.taxiways.forEach(tw => {
      if (!taxiwayShouldDrawInViewport(tw, _twVb)) return;
      const g = taxiwayDrawContext(tw);
      if (!g || tw.vertices.length < 2) return;
      if (interactiveLite && g.isApronTaxiwayPath && !state.isPanning) return;
      const isRunwayPath = g.isRunwayPath, isRunwayExit = g.isRunwayExit, isApronTaxiwayPath = g.isApronTaxiwayPath, sel = g.sel, pathFillWide = g.pathFillWide, pathLineCap = g.pathLineCap, width = g.width;
      const monoLineCss = layerMonoLinesOn() && !sel ? c2dLayerMonoLineStrokeCss() : null;
      if (!state.layers.pathLines) return;
      if (interactiveLite && !isRunwayPath && !state.isPanning) return;
      if (pathFillWide) {
        let skipRunwayCenterlineStroke = false;
        if (isRunwayPath) {
          ctx.lineWidth = centerlineWidthWorld(1.5, 1.4);
          const rwPtsCk = tw.vertices.map(function(v) { return cellToPixel(v.col, v.row); });
          const tLen = runwayPolylineLengthPx(rwPtsCk);
          const rwW = Math.max(24, Number(width) || RUNWAY_PATH_DEFAULT_WIDTH);
          skipRunwayCenterlineStroke = tLen >= Math.max(220, rwW * 3);
          ctx.strokeStyle = monoLineCss || c2dRunwayCenterlineColor();
          ctx.setLineDash([10, 12]);
        } else {
          ctx.setLineDash([]);
        }
        ctx.beginPath();
        for (let i = 0; i < tw.vertices.length; i++) {
          const [x, y] = cellToPixel(tw.vertices[i].col, tw.vertices[i].row);
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        if (!skipRunwayCenterlineStroke) {
          if (isRunwayPath) {
            if (sel) {
              ctx.save();
              ctx.shadowColor = c2dObjectSelectedGlow();
              ctx.shadowBlur = c2dObjectSelectedGlowBlur();
              ctx.stroke();
              ctx.restore();
            } else {
              ctx.stroke();
            }
          } else if (sel) {
            ctx.lineWidth = 0.5;
            ctx.strokeStyle = c2dObjectSelectedStroke();
            ctx.save();
            ctx.shadowColor = c2dObjectSelectedGlow();
            ctx.shadowBlur = c2dObjectSelectedGlowBlur();
            ctx.stroke();
            ctx.restore();
          } else {
            strokeTwCenterlineYellow(monoLineCss || (isRunwayExit ? c2dRunwayTaxiwayCenterlineStroke() : c2dTaxiwayCenterlineStroke()));
          }
        }
        ctx.setLineDash([]);
      } else if (!isRunwayPath) {
        ctx.beginPath();
        for (let i = 0; i < tw.vertices.length; i++) {
          const [x, y] = cellToPixel(tw.vertices[i].col, tw.vertices[i].row);
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        if (sel) {
          ctx.lineWidth = 0.5;
          ctx.strokeStyle = c2dObjectSelectedStroke();
          ctx.stroke();
        } else {
          strokeTwCenterlineYellow(monoLineCss || (isRunwayExit ? c2dRunwayTaxiwayCenterlineStroke() : c2dTaxiwayCenterlineStroke()));
        }
      } else {
        ctx.lineWidth = centerlineWidthWorld(1.5, 1.4);
        ctx.strokeStyle = monoLineCss || c2dRunwayCenterlineColor();
        ctx.setLineDash([10, 12]);
        ctx.lineCap = pathLineCap;
        ctx.lineJoin = 'round';
        ctx.beginPath();
        for (let i = 0; i < tw.vertices.length; i++) {
          const [x, y] = cellToPixel(tw.vertices[i].col, tw.vertices[i].row);
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        if (sel) {
          ctx.save();
          ctx.shadowColor = c2dObjectSelectedGlow();
          ctx.shadowBlur = c2dObjectSelectedGlowBlur();
          ctx.stroke();
          ctx.restore();
        } else ctx.stroke();
        ctx.setLineDash([]);
      }
    });
    state.taxiways.forEach(tw => {
      if (!taxiwayShouldDrawInViewport(tw, _twVb)) return;
      const g = taxiwayDrawContext(tw);
      if (!g || !g.isRunwayPath || tw.vertices.length < 2 || !g.pathFillWide) return;
      if (interactiveLite && !g.sel) return;
      if (_twPanCoarse && !g.sel) return;
      const runwayPts = tw.vertices.map(v => cellToPixel(v.col, v.row));
      drawRunwayPavedCenterlineDashed(tw, runwayPts, g.width);
    });
    state.taxiways.forEach(tw => {
      if (!taxiwayShouldDrawInViewport(tw, _twVb)) return;
      const g = taxiwayDrawContext(tw);
      if (!g || !g.isRunwayPath || tw.vertices.length < 2 || !g.pathFillWide) return;
      if (interactiveLite && !g.sel) return;
      if (_twPanCoarse && !g.sel) return;
      const runwayPts = tw.vertices.map(v => cellToPixel(v.col, v.row));
      drawRunwayDecorations(tw, runwayPts, g.width, { markingsOnly: true });
    });
    state.taxiways.forEach(tw => {
      if (!taxiwayShouldDrawInViewport(tw, _twVb)) return;
      const g = taxiwayDrawContext(tw);
      if (!g) return;
      const drawing = g.drawing, isRunwayPath = g.isRunwayPath, isRunwayExit = g.isRunwayExit, isApronTaxiwayPath = g.isApronTaxiwayPath, width = g.width, sel = g.sel;
      if (!state.layers.pathLines) {
        if (interactiveLite && !drawing && !sel) return;
        if (drawing && tw.vertices.length >= 1) {
          const ptsPx = tw.vertices.map(function(v) { return cellToPixel(v.col, v.row); });
          const ptrTw = state.layoutPathDrawPointer;
          const hoverTw = (ptrTw && ptrTw.length >= 2) ? ptrTw : null;
          strokeLayoutPathDraftPolyline(ctx, ptsPx, hoverTw);
        }
        if (tw.vertices.length >= 1) {
          tw.vertices.forEach((v, i) => {
            const [x, y] = cellToPixel(v.col, v.row);
            const vertexSelected = isSelectedVertex('taxiway', tw.id, i);
            if (drawing) {
              if (i === 0) {
                ctx.fillStyle = '#f97316';
                ctx.beginPath();
                ctx.arc(x, y, c2dPathDrawStartMarkerRadiusPx(), 0, Math.PI*2);
                ctx.fill();
                ctx.strokeStyle = '#ea580c';
                ctx.lineWidth = c2dPathDrawStartMarkerStrokePx();
                ctx.stroke();
                ctx.fillStyle = '#fff';
                ctx.font = 'bold ' + c2dPathDrawStartLabelFontPx() + 'px system-ui';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                if (!interactiveLite) ctx.fillText('Start', x, y + c2dPathDrawStartLabelOffsetY());
              } else {
                layoutMarkerDrawEndpointDot(ctx, x, y, vertexSelected);
              }
            } else if (sel) {
              ctx.fillStyle = vertexSelected ? '#f43f5e' : ((i === 0 && sel) ? '#f97316' : '#e5e7eb');
              ctx.beginPath();
              ctx.arc(x, y, layoutPathVertexRadiusPx(vertexSelected, sel), 0, Math.PI*2);
              ctx.fill();
            }
          });
        }
        return;
      }
      if (interactiveLite && !drawing && !sel) return;
      const dir = getTaxiwayDirection(tw);
      const monoLineCssAr = layerMonoLinesOn() && !sel ? c2dLayerMonoLineStrokeCss() : null;
      if ((!_twPanCoarse || sel || g.drawing) && dir !== 'both' && tw.vertices.length >= 2) {
        const pts = tw.vertices.map(v => cellToPixel(v.col, v.row));
        const totalLen = pts.reduce((acc, p, i) => acc + (i > 0 ? Math.hypot(p[0]-pts[i-1][0], p[1]-pts[i-1][1]) : 0), 0);
        let numArrows;
        if (isRunwayPath) {
          numArrows = Math.max(2, Math.min(5, 1 + Math.floor(totalLen / Math.max(CELL_SIZE * 16, totalLen / 6))));
        } else {
          const arrowSpacing = Math.max(22, Math.min(42, totalLen / 10));
          numArrows = Math.max(2, Math.floor(totalLen / arrowSpacing));
        }
        const arrLen = isRunwayPath ? CELL_SIZE * 0.63 : CELL_SIZE * 0.54;
        ctx.fillStyle = monoLineCssAr || (isRunwayPath ? c2dRunwayCenterlineColor() : (isRunwayExit ? c2dRunwayTaxiwayCenterlineStroke() : c2dTaxiwayCenterlineStroke()));
        for (let k = 1; k <= numArrows; k++) {
          const targetDist = totalLen * (k / (numArrows + 1));
          let acc = 0;
          let ax = pts[0][0], ay = pts[0][1];
          let angle = Math.atan2(pts[1][1]-pts[0][1], pts[1][0]-pts[0][0]);
          for (let i = 1; i < pts.length; i++) {
            const seg = Math.hypot(pts[i][0]-pts[i-1][0], pts[i][1]-pts[i-1][1]);
            angle = Math.atan2(pts[i][1]-pts[i-1][1], pts[i][0]-pts[i-1][0]);
            if (acc + seg >= targetDist) {
              const t = seg > 0 ? (targetDist - acc) / seg : 0;
              ax = pts[i-1][0] + t * (pts[i][0]-pts[i-1][0]);
              ay = pts[i-1][1] + t * (pts[i][1]-pts[i-1][1]);
              break;
            }
            acc += seg;
          }
          if (dir === 'counter_clockwise' && !isRunwayPath) angle += Math.PI;
          ctx.beginPath();
          ctx.moveTo(ax + arrLen * Math.cos(angle), ay + arrLen * Math.sin(angle));
          ctx.lineTo(ax - arrLen * 0.7 * Math.cos(angle) + arrLen * 0.4 * Math.sin(angle), ay - arrLen * 0.7 * Math.sin(angle) - arrLen * 0.4 * Math.cos(angle));
          ctx.lineTo(ax - arrLen * 0.7 * Math.cos(angle) - arrLen * 0.4 * Math.sin(angle), ay - arrLen * 0.7 * Math.sin(angle) + arrLen * 0.4 * Math.cos(angle));
          ctx.closePath();
          ctx.fill();
        }
      }
      if (!interactiveLite && (!_twPanCoarse || sel || g.drawing) && isRunwayPath && tw.vertices.length >= 2) {
        const rwPts = tw.vertices.map(function(v) { return cellToPixel(v.col, v.row); });
        if (rwPts.length >= 2) {
          const lenPx = runwayPolylineLengthPx(rwPts);
          const d = getEffectiveRunwayLineupDistFromStartM(tw, lenPx);
          const lp = _pointOnPolylineAtDistPxForLineup(rwPts, d);
          if (lp) {
            const lineupRtxOk = isLineupPointTouchingRunwayTaxiwayOnRunway(tw, lp);
            ctx.save();
            ctx.fillStyle = lineupRtxOk ? '#16a34a' : '#dc2626';
            ctx.strokeStyle = lineupRtxOk ? '#14532d' : '#450a0a';
            ctx.lineWidth = 1.2;
            ctx.beginPath();
            ctx.arc(lp[0], lp[1], 5 * LAYOUT_VERTEX_DOT_SCALE, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();
            const labelText = 'Line up';
            ctx.font = 'bold 11px system-ui, sans-serif';
            const padX = 6, padY = 4, rad = 5;
            const mLabel = ctx.measureText(labelText);
            const bw = mLabel.width + padX * 2;
            const bh = 11 + padY * 2;
            const bx = lp[0] + 7;
            const by = lp[1] - 4 - bh;
            ctx.beginPath();
            if (typeof ctx.roundRect === 'function') ctx.roundRect(bx, by, bw, bh, rad);
            else ctx.rect(bx, by, bw, bh);
            ctx.fillStyle = lineupRtxOk ? 'rgba(22, 163, 74, 0.92)' : 'rgba(220, 38, 38, 0.92)';
            ctx.fill();
            ctx.strokeStyle = lineupRtxOk ? '#14532d' : '#450a0a';
            ctx.lineWidth = 1.2;
            ctx.stroke();
            ctx.fillStyle = '#ffffff';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(labelText, bx + bw / 2, by + bh / 2);
            ctx.restore();
            const hop1 = listRtxTouchingLineupOnRunway(tw, lp);
            for (let hi = 0; hi < hop1.length; hi++) {
              const rtx = hop1[hi];
              if (!rtx) continue;
              const nid = rtxRunwayExitNeighborIds(rtx);
              if (typeof rtxSetHasRunwayHoldingHp === 'function' && rtxSetHasRunwayHoldingHp(nid)) continue;
              const vts = rtx.vertices || [];
              if (vts.length < 2) continue;
              let sx = 0, sy = 0;
              for (let vi = 0; vi < vts.length; vi++) {
                const pp = cellToPixel(vts[vi].col, vts[vi].row);
                sx += pp[0]; sy += pp[1];
              }
              const mx = sx / vts.length, my = sy / vts.length;
              const badgeText = 'No Holding Point';
              ctx.save();
              ctx.font = 'bold 10px system-ui, sans-serif';
              const padXB = 6, padYB = 3, radB = 4;
              const mw = ctx.measureText(badgeText).width + padXB * 2;
              const mh = 10 + padYB * 2;
              const bxx = mx - mw / 2, byy = my - 22;
              ctx.beginPath();
              if (typeof ctx.roundRect === 'function') ctx.roundRect(bxx, byy, mw, mh, radB);
              else ctx.rect(bxx, byy, mw, mh);
              ctx.fillStyle = 'rgba(220, 38, 38, 0.95)';
              ctx.fill();
              ctx.strokeStyle = '#450a0a';
              ctx.lineWidth = 1.1;
              ctx.stroke();
              ctx.fillStyle = '#ffffff';
              ctx.textAlign = 'center';
              ctx.textBaseline = 'middle';
              ctx.fillText(badgeText, bxx + mw / 2, byy + mh / 2);
              ctx.restore();
            }
          }
        }
      }
      if (drawing && tw.vertices.length >= 1) {
        const ptsPx = tw.vertices.map(function(v) { return cellToPixel(v.col, v.row); });
        const ptrTw = state.layoutPathDrawPointer;
        const hoverTw = (ptrTw && ptrTw.length >= 2) ? ptrTw : null;
        strokeLayoutPathDraftPolyline(ctx, ptsPx, hoverTw);
      }
      if (tw.vertices.length >= 1) {
        tw.vertices.forEach((v, i) => {
          const [x, y] = cellToPixel(v.col, v.row);
          const vertexSelected = isSelectedVertex('taxiway', tw.id, i);
          if (drawing) {
            if (i === 0) {
              ctx.fillStyle = '#f97316';
              ctx.beginPath();
              ctx.arc(x, y, c2dPathDrawStartMarkerRadiusPx(), 0, Math.PI*2);
              ctx.fill();
              ctx.strokeStyle = '#ea580c';
              ctx.lineWidth = c2dPathDrawStartMarkerStrokePx();
              ctx.stroke();
              ctx.fillStyle = '#fff';
              ctx.font = 'bold ' + c2dPathDrawStartLabelFontPx() + 'px system-ui';
              ctx.textAlign = 'center';
              ctx.textBaseline = 'middle';
              if (!interactiveLite) ctx.fillText('Start', x, y + c2dPathDrawStartLabelOffsetY());
            } else {
              layoutMarkerDrawEndpointDot(ctx, x, y, vertexSelected);
            }
          } else if (sel) {
            ctx.fillStyle = vertexSelected ? '#f43f5e' : ((i === 0 && sel) ? '#f97316' : '#e5e7eb');
            ctx.beginPath();
            ctx.arc(x, y, layoutPathVertexRadiusPx(vertexSelected, sel), 0, Math.PI*2);
            ctx.fill();
          }
        });
      }
    });
    ctx.restore();
  }

  function drawApronTaxiwayLinks() {
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.translate(state.panX, state.panY);
    ctx.scale(state.scale, state.scale);
    state.apronLinks.forEach(lk => {
      const stand = findStandById(lk.pbbId);
      const tw = state.taxiways.find(t => t.id === lk.taxiwayId);
      if (!stand || !tw || lk.tx == null || lk.ty == null) return;
      const poly = getApronLinkPolylineWorldPts(lk);
      if (poly.length < 2) return;
      ctx.setLineDash([]);
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';
      function traceApronLinkPoly() {
        ctx.moveTo(poly[0][0], poly[0][1]);
        for (let pi = 1; pi < poly.length; pi++) ctx.lineTo(poly[pi][0], poly[pi][1]);
      }
      if (state.layers.pathLines) {
        ctx.beginPath();
        traceApronLinkPoly();
        ctx.lineWidth = 1;
        ctx.strokeStyle = layerMonoLinesOn() ? c2dLayerMonoLineStrokeCss() : c2dTaxiwayCenterlineStroke();
        ctx.stroke();
      }
      const svApron = state.selectedVertex;
      const selApron = state.selectedObject && state.selectedObject.type === 'apronLink' && state.selectedObject.id === lk.id;
      if (selApron) {
        ctx.setLineDash([]);
        const standPxLoop = getApronLinkStandEndPx(lk);
        const twEndPx = [Number(lk.tx), Number(lk.ty)];
        const vtxMatchD2 = (CELL_SIZE * HIT_TW_VTX_CF) ** 2;
        for (let pi = 0; pi < poly.length; pi++) {
          const [px, py] = poly[pi];
          let isStandEnd = !!(standPxLoop && dist2([px, py], standPxLoop) <= vtxMatchD2);
          let isTaxiEnd = isFinite(twEndPx[0]) && isFinite(twEndPx[1]) && dist2([px, py], twEndPx) <= vtxMatchD2;
          let midIdx = -1;
          if (!isStandEnd && !isTaxiEnd) {
            (lk.midVertices || []).forEach(function(v, mi) {
              const mpx = v && isFinite(Number(v.x)) && isFinite(Number(v.y))
                ? [Number(v.x), Number(v.y)]
                : cellToPixel(Number(v.col), Number(v.row));
              if (dist2([px, py], mpx) <= vtxMatchD2) midIdx = mi;
            });
          }
          let vtxSel = false;
          let draggable = false;
          if (isTaxiEnd) {
            draggable = true;
            vtxSel = !!(svApron && svApron.type === 'apronLink' && svApron.id === lk.id && svApron.kind === 'taxiway');
          } else if (midIdx >= 0) {
            draggable = true;
            vtxSel = !!(svApron && svApron.type === 'apronLink' && svApron.id === lk.id && svApron.kind === 'mid' && svApron.midIndex === midIdx);
          }
          const r = layoutPathVertexRadiusPx(vtxSel, draggable);
          ctx.fillStyle = vtxSel
            ? '#f43f5e'
            : (layerMonoLinesOn()
              ? (draggable ? '#cbd5e1' : c2dLayerMonoLineStrokeCss())
              : (draggable ? '#86efac' : '#22c55e'));
          ctx.beginPath();
          ctx.arc(px, py, r, 0, Math.PI*2);
          ctx.fill();
        }
      }
    });
    ctx.setLineDash([]);
    if (state.apronLinkTemp && (state.layers.pathLines || state.apronLinkDrawing)) {
      const t = state.apronLinkTemp;
      const ptsPx = [];
      if (t.kind === 'pbb' || t.kind === 'remote') {
        const st = findStandById(t.standId);
        if (st) ptsPx.push(getStandApronTaxiwayAttachWorldPx(st));
      } else if (t.kind === 'taxiway') {
        ptsPx.push([t.x, t.y]);
      }
      (state.apronLinkMidpoints || []).forEach(function(c) {
        if (c && isFinite(Number(c.x)) && isFinite(Number(c.y))) ptsPx.push([Number(c.x), Number(c.y)]);
        else ptsPx.push(cellToPixel(Number(c.col), Number(c.row)));
      });
      const hoverApron = (state.apronLinkPointerWorld && state.apronLinkPointerWorld.length >= 2) ? state.apronLinkPointerWorld : null;
      if (ptsPx.length >= 1) {
        strokeLayoutPathDraftPolyline(ctx, ptsPx, hoverApron);
        drawLayoutPathDraftVertexDots(ctx, ptsPx, hoverApron);
      }
    }
    ctx.restore();
  }

  function flightTimelineSegmentAtSimTime(flight, tSec) {
    const tl = flight && flight.timeline;
    if (!tl || tl.length < 2) return null;
    let t = Number(tSec);
    if (!isFinite(t)) return null;
    if (t + 1e-9 < tl[0].t) return null;
    if (t > tl[tl.length - 1].t) t = tl[tl.length - 1].t;
    for (let i = 0; i < tl.length - 1; i++) {
      const a = tl[i], b = tl[i + 1];
      if (t >= a.t && t <= b.t) return { a: a, b: b };
    }
    return null;
  }
  function isTimelineSegmentStationaryWorld(a, b) {
    const dx = b.x - a.x, dy = b.y - a.y;
    return dx * dx + dy * dy < 0.64;
  }
  function countFlightsWaitingAtHoldingPoint2D(hp, tSec) {
    if (!hp || !isFinite(hp.x) || !isFinite(hp.y)) return 0;
    if (!state.hasSimulationResult) return 0;
    if (typeof getFlightPoseAtTimeForDraw !== 'function') return 0;
    const t = Number(tSec);
    if (!isFinite(t)) return 0;
    const hx = hp.x, hy = hp.y;
    const dia = typeof c2dHoldingPointDiameterM === 'function' ? c2dHoldingPointDiameterM() : 24;
    const rad = Math.max(10, dia * 0.55);
    const rad2 = rad * rad;
    let n = 0;
    const flights = state.flights || [];
    for (let i = 0; i < flights.length; i++) {
      const f = flights[i];
      if (!f || flightBlockedLikeNoWay(f)) continue;
      const pose = getFlightPoseAtTimeForDraw(f, t);
      if (!pose) continue;
      const dx = pose.x - hx, dy = pose.y - hy;
      if (dx * dx + dy * dy > rad2) continue;
      const seg = flightTimelineSegmentAtSimTime(f, t);
      if (!seg || !isTimelineSegmentStationaryWorld(seg.a, seg.b)) continue;
      n++;
    }
    return n;
  }
  function firstFlightWaitingAtHoldingPoint2D(hp, tSec) {
    if (!hp || !isFinite(hp.x) || !isFinite(hp.y)) return null;
    if (!state.hasSimulationResult) return null;
    if (typeof getFlightPoseAtTimeForDraw !== 'function') return null;
    const t = Number(tSec);
    if (!isFinite(t)) return null;
    const hx = hp.x, hy = hp.y;
    const dia = typeof c2dHoldingPointDiameterM === 'function' ? c2dHoldingPointDiameterM() : 24;
    const rad = Math.max(10, dia * 0.55);
    const rad2 = rad * rad;
    const flights = state.flights || [];
    for (let i = 0; i < flights.length; i++) {
      const f = flights[i];
      if (!f || flightBlockedLikeNoWay(f)) continue;
      const pose = getFlightPoseAtTimeForDraw(f, t);
      if (!pose) continue;
      const dx = pose.x - hx, dy = pose.y - hy;
      if (dx * dx + dy * dy > rad2) continue;
      const seg = flightTimelineSegmentAtSimTime(f, t);
      if (!seg || !isTimelineSegmentStationaryWorld(seg.a, seg.b)) continue;
      return f;
    }
    return null;
  }
  function polylineTangentForwardAtDistance(pts, sAlong) {
    if (!pts || pts.length < 2) return [1, 0];
    if (typeof polylineTotalLength !== 'function' || typeof polylinePointAtDistance !== 'function') return [1, 0];
    const total = polylineTotalLength(pts);
    if (total < 1e-6) return [1, 0];
    const eps = 2;
    const s0 = Math.max(0, Math.min(Number(sAlong) || 0, total));
    let s1 = Math.min(s0 + eps, total);
    let p0 = polylinePointAtDistance(pts, s0);
    let p1 = polylinePointAtDistance(pts, s1);
    let dx = p1[0] - p0[0], dy = p1[1] - p0[1];
    if (dx * dx + dy * dy < 1e-10) {
      s1 = Math.max(0, s0 - eps);
      p1 = polylinePointAtDistance(pts, s0);
      p0 = polylinePointAtDistance(pts, s1);
      dx = p1[0] - p0[0];
      dy = p1[1] - p0[1];
    }
    const len = Math.hypot(dx, dy) || 1;
    return [dx / len, dy / len];
  }
  function drawHoldingQueueGhostFlights2D() {
    if (!ctx) return;
    if (!state.hasSimulationResult) return;
    if (!state.flights || !state.flights.length) return;
    if (typeof getFlightPoseAtTimeForDraw !== 'function') return;
    if (typeof graphPathDeparture !== 'function' || typeof cumulativeDistAlongPolylineToPoint !== 'function') return;
    if (typeof polylinePointAtDistance !== 'function' || typeof polylineTotalLength !== 'function') return;
    const tSecDraw = state.simTimeSec;
    const HOLDING_QUEUE_GHOST_SPACING_M = 70;
    const dia = typeof c2dHoldingPointDiameterM === 'function' ? c2dHoldingPointDiameterM() : 24;
    const rad = Math.max(10, dia * 0.55);
    const pathTol2 = Math.pow(Math.max(rad * 4, 45), 2);
    const silN = Number(_acSil.noseX), silWR = Number(_acSil.wingRearX), silUY = Number(_acSil.wingUpperY);
    const silTN = Number(_acSil.tailNeckX), silLY = Number(_acSil.wingLowerY);
    const nX = isFinite(silN) ? silN : 0.6;
    const wRx = isFinite(silWR) ? silWR : -0.5;
    const uY = isFinite(silUY) ? silUY : 0.35;
    const tX = isFinite(silTN) ? silTN : -0.3;
    const lY = isFinite(silLY) ? silLY : -0.35;
    const useDetailSil = _ac2d.useDetailedSilhouette === true;
    const silhouette2D = getApronAircraftDetailedSilhouettePoints();
    const outW = Number(_ac2d.outlineWidth);
    const outlineWidth = (isFinite(outW) && outW > 0) ? outW : 0;
    const outlineColor = _ac2d.outlineColor || '';
    const fcModeG = state.flightColorMode || 'all';
    const fcPalG = flightSimVizPaletteList();
    const fcOverG = flightSimVizOverflowGray();
    const fcKeyIdxG = buildFlightSim2DColorKeyIndexMap();
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.translate(state.panX, state.panY);
    ctx.scale(state.scale, state.scale);
    (state.holdingPoints || []).forEach(function(hp) {
      if (!hp || !isFinite(hp.x) || !isFinite(hp.y)) return;
      const waitN = countFlightsWaitingAtHoldingPoint2D(hp, tSecDraw);
      if (waitN < 2) return;
      const f = firstFlightWaitingAtHoldingPoint2D(hp, tSecDraw);
      if (!f) return;
      const dimsMG = getSimAircraftWorldDimsM(f);
      const lenMG = dimsMG.lenM, wingMG = dimsMG.wingM;
      let scaleX, scaleY;
      if (useDetailSil && silhouette2D.length >= 3) {
        const sp = detailedSilhouetteAxisSpans(silhouette2D);
        scaleX = lenMG / sp.spanX;
        scaleY = wingMG / sp.spanY;
      } else {
        const xs = [nX, wRx, tX];
        const minXn = Math.min(xs[0], xs[1], xs[2]);
        const maxXn = Math.max(xs[0], xs[1], xs[2]);
        const lenNorm = Math.max(1e-9, maxXn - minXn);
        const wingNorm = Math.max(1e-9, uY + lY);
        scaleX = lenMG / lenNorm;
        scaleY = wingMG / wingNorm;
      }
      const pts = graphPathDeparture(f, { onlyToLineup: true });
      if (!pts || pts.length < 2) return;
      const cum = cumulativeDistAlongPolylineToPoint(pts, [hp.x, hp.y]);
      if (!cum || cum.d2 > pathTol2) return;
      const sHp = cum.distAlong;
      for (let k = 1; k < waitN; k++) {
        const s = sHp - k * HOLDING_QUEUE_GHOST_SPACING_M;
        if (s < -0.5) break;
        const sDraw = Math.max(0, s);
        const pt = polylinePointAtDistance(pts, sDraw);
        const tan = polylineTangentForwardAtDistance(pts, sDraw);
        const nx = tan[0], ny = tan[1];
        ctx.save();
        ctx.translate(pt[0], pt[1]);
        ctx.rotate(Math.atan2(ny, nx));
        ctx.fillStyle = resolveFlightSim2DGlyphFillRgba(f, false, fcKeyIdxG, fcPalG, fcOverG, fcModeG);
        ctx.beginPath();
        if (useDetailSil) {
          ctx.moveTo(silhouette2D[0][0] * scaleX, silhouette2D[0][1] * scaleY);
          for (let si = 1; si < silhouette2D.length; si++) ctx.lineTo(silhouette2D[si][0] * scaleX, silhouette2D[si][1] * scaleY);
          ctx.closePath();
        } else {
          ctx.moveTo(scaleX * nX, 0);
          ctx.lineTo(scaleX * wRx, scaleY * uY);
          ctx.lineTo(scaleX * tX, 0);
          ctx.lineTo(scaleX * wRx, scaleY * lY);
          ctx.closePath();
        }
        ctx.fill();
        if (outlineWidth > 0 && outlineColor) {
          ctx.strokeStyle = outlineColor;
          ctx.lineWidth = outlineWidth;
          ctx.stroke();
        } else if (useDetailSil) {
          ctx.strokeStyle = 'rgba(15,23,42,1)';
          ctx.lineWidth = 1.1;
          ctx.stroke();
        }
        ctx.restore();
      }
    });
    ctx.restore();
  }
  function drawHoldingPoints2D(interactiveLite) {
    if (!ctx) return;
    if (interactiveLite) return;
    const vb = layoutWorldViewportAabbWithBufferM(LAYOUT_RENDER_VIEWPORT_BUFFER_M);
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.translate(state.panX, state.panY);
    ctx.scale(state.scale, state.scale);
    const sel = state.selectedObject && state.selectedObject.type === 'holdingPoint';
    if (!state.layers.pathLines && !state.holdingPointDrawing && !sel) {
      ctx.restore();
      return;
    }
    (state.holdingPoints || []).forEach(function(hp) {
      if (!hp || !isFinite(hp.x) || !isFinite(hp.y)) return;
      const hpAabb = { minX: hp.x - CELL_SIZE, minY: hp.y - CELL_SIZE, maxX: hp.x + CELL_SIZE, maxY: hp.y + CELL_SIZE };
      if (!aabbIntersectsViewport(vb, hpAabb)) return;
      const selected = sel && state.selectedObject.id === hp.id;
      drawHoldingPointGridMarking(ctx, hp.x, hp.y, hp.hpKind, selected, false);
      const waitN = countFlightsWaitingAtHoldingPoint2D(hp, state.simTimeSec);
      if (waitN > 0 && !interactiveLite) {
        const tt = findHoldingPointPathGeometry(hp);
        const bump = Math.max(12, (Number(tt.pathWidthM) || 0) * 0.42);
        const bx = hp.x + tt.ux * bump;
        const by = hp.y + tt.uy * bump;
        const label = String(waitN);
        const fs = Math.max(9, Math.min(15, 11 / Math.max(0.22, state.scale)));
        ctx.font = 'bold ' + fs + 'px system-ui, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        const tw = ctx.measureText(label).width;
        const padX = fs * 0.42;
        const padY = fs * 0.28;
        const bw = tw + padX * 2;
        const bh = fs + padY * 2;
        const left = bx - bw / 2;
        const top = by - bh / 2;
        const rr = Math.min(bh * 0.45, fs * 0.5);
        ctx.beginPath();
        ctx.moveTo(left + rr, top);
        ctx.lineTo(left + bw - rr, top);
        ctx.quadraticCurveTo(left + bw, top, left + bw, top + rr);
        ctx.lineTo(left + bw, top + bh - rr);
        ctx.quadraticCurveTo(left + bw, top + bh, left + bw - rr, top + bh);
        ctx.lineTo(left + rr, top + bh);
        ctx.quadraticCurveTo(left, top + bh, left, top + bh - rr);
        ctx.lineTo(left, top + rr);
        ctx.quadraticCurveTo(left, top, left + rr, top);
        ctx.closePath();
        if (layerMonoLinesOn()) {
          ctx.fillStyle = c2dLayerMonoFillDarkAsphaltRgba(0.92);
          ctx.strokeStyle = c2dLayerMonoLineStrokeCss();
        } else {
          ctx.fillStyle = 'rgba(15, 23, 42, 0.94)';
          ctx.strokeStyle = 'rgba(148, 163, 184, 0.95)';
        }
        ctx.lineWidth = Math.max(0.75, 1.15 / Math.max(state.scale, 0.08));
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = layerMonoEtcOn() ? C2D_LAYER_MONO_ETC_WHITE : '#f1f5f9';
        ctx.fillText(label, bx, by);
      }
    });
    if (state.holdingPointDrawing && state.previewHoldingPoint) {
      const px = state.previewHoldingPoint.x, py = state.previewHoldingPoint.y;
      const ptp = state.previewHoldingPoint.pathType || 'taxiway';
      drawHoldingPointGridMarking(ctx, px, py, pathTypeToHpKind(ptp), false, true);
    }
    ctx.restore();
  }

  function drawStandPreview(interactiveLite) {
    const nowPerf = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
    const suppressStandFill = !!state.isPanning || nowPerf < _layoutDetailSuppressUntil;
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.translate(state.panX, state.panY);
    ctx.scale(state.scale, state.scale);
    const mode = settingModeSelect.value;
    if (mode === 'remote' && state.previewRemote) {
      const cx = Number(state.previewRemote.x), cy = Number(state.previewRemote.y);
      const category = panelRepresentativeCategoryForNewStand('remote');
      const depPr = getStandDepthMeters(category);
      const widPr = getStandWidthMeters(category);
      const angle = 0;
      const overlap = state.previewRemote.overlap;
      ctx.fillStyle = overlap ? 'rgba(239,68,68,0.35)' : 'rgba(34,197,94,0.25)';
      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(angle);
      ctx.setLineDash([]);
      if (!interactiveLite && !suppressStandFill) fillStandSafetyFootprintInLocalAxes(ctx, depPr, widPr, category);
      drawStandSafetyContourInLocalAxes(ctx, depPr, widPr, category, false);
      if (!interactiveLite) drawStandApronMarkingsInLocalAxes(ctx, depPr, widPr, category);
      ctx.restore();
    }
    if (mode === 'tempStand' && state.previewTempStand) {
      const cx = Number(state.previewTempStand.x), cy = Number(state.previewTempStand.y);
      const category = panelRepresentativeCategoryForNewStand('tempStand');
      const depPt = getStandDepthMeters(category);
      const widPt = getStandWidthMeters(category);
      const angle = 0;
      const overlap = state.previewTempStand.overlap;
      ctx.fillStyle = overlap ? 'rgba(239,68,68,0.35)' : 'rgba(167,139,250,0.28)';
      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(angle);
      ctx.setLineDash([]);
      if (!interactiveLite && !suppressStandFill) fillStandSafetyFootprintInLocalAxes(ctx, depPt, widPt, category);
      drawStandSafetyContourInLocalAxes(ctx, depPt, widPt, category, false);
      if (!interactiveLite) drawStandApronMarkingsInLocalAxes(ctx, depPt, widPt, category);
      ctx.restore();
      ctx.setLineDash([]);
      const pjr = Math.max(3, 3.5 / Math.max(state.scale, 0.08));
      ctx.strokeStyle = '#22d3ee';
      ctx.lineWidth = 1.75;
      ctx.beginPath();
      ctx.arc(cx, cy, pjr, 0, Math.PI * 2);
      ctx.stroke();
      ctx.fillStyle = 'rgba(236,254,255,0.85)';
      ctx.beginPath();
      ctx.arc(cx, cy, pjr * 0.45, 0, Math.PI * 2);
      ctx.fill();
    }
    if (mode === 'pbb' && state.previewPbb) {
      const pv = state.previewPbb;
      let ex = Number(pv.x2), ey = Number(pv.y2);
      if (pv.apronSiteX != null && pv.apronSiteY != null) {
        const ax = Number(pv.apronSiteX), ay = Number(pv.apronSiteY);
        if (Number.isFinite(ax) && Number.isFinite(ay)) { ex = ax; ey = ay; }
      }
      const catPv = state.previewPbb.category || 'C';
      const depPv = getStandDepthMeters(catPv);
      const widPv = getStandWidthMeters(catPv);
      const overlap = state.previewPbb.overlap;
      const warnOuterOverlap = !!state.previewPbb.warnOuterOverlap;
      const angle = getPBBStandAngle(state.previewPbb);
      ctx.fillStyle = overlap
        ? 'rgba(239,68,68,0.35)'
        : (warnOuterOverlap ? 'rgba(245,158,11,0.30)' : 'rgba(34,197,94,0.25)');
      ctx.save();
      ctx.translate(ex, ey);
      ctx.rotate(angle);
      ctx.setLineDash([]);
      if (!interactiveLite && !suppressStandFill) fillStandSafetyFootprintInLocalAxes(ctx, depPv, widPv, catPv);
      drawStandSafetyContourInLocalAxes(ctx, depPv, widPv, catPv, false);
      if (!interactiveLite) drawStandApronMarkingsInLocalAxes(ctx, depPv, widPv, catPv);
      if (!interactiveLite) {
        ctx.fillStyle = '#bbf7d0';
        ctx.font = '10px system-ui';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(state.previewPbb.category || panelRepresentativeCategoryForNewStand('pbb'), 0, 0);
      }
      ctx.restore();
    }
    ctx.restore();
  }

  let _safeDrawErrLogged = false;
  let _drawRafId = 0;
  /** While panning or shortly after wheel zoom, skip heavy path layers for smoother interaction. */
  let _layoutDetailSuppressUntil = 0;
  /** Offscreen bitmap: grid → holding points (no sim-time stand fill / flights). Used during sim playback. */
  let _simPlaybackBgCanvas = null;
  let _simPlaybackBgSig = '';
  /** Baked heatmap in world space; redraw bitmap only when sim/phase/layout/tClip changes. */
  let _layoutHeatmapSvg = null;
  let _layoutHeatmapSvgG = null;
  let _layoutHeatmapSvgContentSig = '';
  let _layoutHeatmapBakedGraphHash = null;
  function ensureSimPlaybackBgCanvasBuffer(w, h) {
    if (!_simPlaybackBgCanvas) _simPlaybackBgCanvas = document.createElement('canvas');
    if (_simPlaybackBgCanvas.width !== w || _simPlaybackBgCanvas.height !== h) {
      _simPlaybackBgCanvas.width = w;
      _simPlaybackBgCanvas.height = h;
      _simPlaybackBgSig = '';
    }
    return _simPlaybackBgCanvas;
  }
  function simPlaybackBackgroundCacheSignature(interactiveLite) {
    if (!layoutDrawCanvas) return '';
    const w = layoutDrawCanvas.width, h = layoutDrawCanvas.height;
    const sel = state.selectedObject;
    const selKey = sel ? (String(sel.type) + ':' + String(sel.id)) : '';
    const layers = state.layers || {};
    const layerKey = Object.keys(layers).sort().map(function(k) { return k + '=' + (layers[k] ? '1' : '0'); }).join('&');
    const lm = state.layerMono || {};
    const layerMonoKey = ['lines', 'fill', 'etc'].map(function(k) { return k + '=' + (lm[k] ? '1' : '0'); }).join('&');
    return [
      w, h, dpr, state.panX, state.panY, state.scale,
      interactiveLite ? '1' : '0',
      selKey,
      state.pathPolylineCacheRev | 0,
      String(state.currentLayoutName || ''),
      layerKey,
      layerMonoKey,
      state.pathArcDrag ? '1' : '0',
      layoutViewIsDragging() ? '1' : '0',
    ].join('|');
  }
  const LAYOUT_DRAG_LITE_VIEWPORT_WIDTH_THRESHOLD_M = 1200;
  function layoutViewportWidthWorldM() {
    if (!layoutDrawCanvas) return Infinity;
    const s = Math.max(state.scale || 1, 1e-9);
    return (layoutDrawCanvas.width / dpr) / s;
  }
  function layoutViewIsDragging() {
    return !!(
      state.pathArcDrag ||
      state.dragVertex ||
      state.dragTaxiwayVertex ||
      state.dragStandRotation ||
      state.dragPbbBridgeVertex ||
      state.dragStandConnection ||
      state.dragRemoteStandPosition ||
      state.dragApronLinkVertex ||
      state.dragLayoutMarkerHandle
    );
  }
  function layoutViewSkipsTaxiDetail(drawOpts) {
    if (drawOpts && drawOpts.forceFullLayoutDraw) return false;
    const now = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
    const wideViewport = layoutViewportWidthWorldM() > LAYOUT_DRAG_LITE_VIEWPORT_WIDTH_THRESHOLD_M;
    if (!wideViewport) return false;
    return !!state.isPanning || layoutViewIsDragging() || now < _layoutDetailSuppressUntil;
  }
  function safeDraw(drawOpts) { try { draw(drawOpts); _safeDrawErrLogged = false; } catch(e) { if (!_safeDrawErrLogged) { console.error('safeDraw: draw() error', e); _safeDrawErrLogged = true; } } }
  function flushDrawNow() {
    if (_drawRafId) {
      cancelAnimationFrame(_drawRafId);
      _drawRafId = 0;
    }
    safeDraw();
  }
  function scheduleDraw() {
    if (_drawRafId) return;
    _drawRafId = requestAnimationFrame(function() {
      _drawRafId = 0;
      safeDraw();
    });
  }
  const LAYOUT_ISLAND_WIDTH_DEFAULT_M = 2;
  const LAYOUT_ISLAND_WIDTH_MAX_M = 200;
  /** Contour (legacy 'island') stroke — bright neon sky cyan. */
  const LAYOUT_ISLAND_STROKE_CSS = '#22e8ff';
  function islandWidthMResolved(m) {
    const v = Number(m && m.widthM);
    if (isFinite(v) && v >= 0) return Math.min(LAYOUT_ISLAND_WIDTH_MAX_M, v);
    const legacy = Number(m && m.outerWidthM);
    if (isFinite(legacy) && legacy >= 0) return Math.min(LAYOUT_ISLAND_WIDTH_MAX_M, legacy);
    return LAYOUT_ISLAND_WIDTH_DEFAULT_M;
  }
  function getMarkerIslandWidthMFromPanel() {
    const el = document.getElementById('markerIslandOuterWidthM');
    const v = Number(el && el.value);
    if (isFinite(v) && v >= 0) return Math.min(LAYOUT_ISLAND_WIDTH_MAX_M, v);
    return LAYOUT_ISLAND_WIDTH_DEFAULT_M;
  }
  function syncPathPavementRadiosToValue(pavement) {
    const p = (pavement === 'cement') ? 'cement' : 'asphalt';
    const el = document.getElementById('pathPavement');
    if (!el || document.activeElement === el) return;
    el.value = p;
  }
  function getPathPavementFromPanelForPathType(pathType) {
    const el = document.getElementById('pathPavement');
    const v = el && el.value;
    if (v === 'asphalt' || v === 'cement') return v;
    return pathPavementDefaultForPathType(pathType);
  }
  function syncMarkerIslandSidebarWidthsFromSelection() {
    const oEl = document.getElementById('markerIslandOuterWidthM');
    if (!oEl) return;
    const so = state.selectedObject;
    if (!so || so.type !== 'layoutMarker' || !so.obj || so.obj.kind !== 'island') return;
    const w = islandWidthMResolved(so.obj);
    if (document.activeElement !== oEl) oEl.value = String(w);
  }
  (function setupMarkerIslandPanelWidthApplyToSelection() {
    const oEl = document.getElementById('markerIslandOuterWidthM');
    function applyWidthToSelectedIsland() {
      const so = state.selectedObject;
      if (!so || so.type !== 'layoutMarker' || !so.obj || so.obj.kind !== 'island') return;
      const v = Number(oEl && oEl.value);
      so.obj.widthM = (isFinite(v) && v >= 0) ? Math.min(LAYOUT_ISLAND_WIDTH_MAX_M, v) : LAYOUT_ISLAND_WIDTH_DEFAULT_M;
      scheduleDraw();
    }
    if (oEl) oEl.addEventListener('input', applyWidthToSelectedIsland);
  })();
  function layoutIslandWorldPointsForDraw(m) {
    if (!m || m.kind !== 'island' || !Array.isArray(m.points)) return [];
    return m.points.map(function(p) { return [Number(p.x), Number(p.y)]; }).filter(function(P) { return isFinite(P[0]) && isFinite(P[1]); });
  }
  function islandInwardUnitNormalForDraw(p0, p1, centroidX, centroidY) {
    const dx = p1[0] - p0[0], dy = p1[1] - p0[1];
    const len = Math.hypot(dx, dy) || 1;
    let nx = -dy / len, ny = dx / len;
    const mx = (p0[0] + p1[0]) * 0.5, my = (p0[1] + p1[1]) * 0.5;
    if ((centroidX - mx) * nx + (centroidY - my) * ny < 0) {
      nx = -nx;
      ny = -ny;
    }
    return [nx, ny];
  }
  function islandLineIntersection(a1, a2, b1, b2) {
    const x1 = a1[0], y1 = a1[1], x2 = a2[0], y2 = a2[1];
    const x3 = b1[0], y3 = b1[1], x4 = b2[0], y4 = b2[1];
    const den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
    if (Math.abs(den) < 1e-14) return null;
    const t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den;
    return [x1 + t * (x2 - x1), y1 + t * (y2 - y1)];
  }
  function islandOffsetPolygonCorners(pts, cx, cy, dist, outward) {
    const n = pts.length;
    const out = [];
    for (let i = 0; i < n; i++) {
      const pPrev = pts[(i - 1 + n) % n];
      const p0 = pts[i];
      const pNext = pts[(i + 1) % n];
      const in1 = islandInwardUnitNormalForDraw(pPrev, p0, cx, cy);
      const in2 = islandInwardUnitNormalForDraw(p0, pNext, cx, cy);
      const o1 = outward ? [-in1[0], -in1[1]] : [in1[0], in1[1]];
      const o2 = outward ? [-in2[0], -in2[1]] : [in2[0], in2[1]];
      const A1 = [pPrev[0] + o1[0] * dist, pPrev[1] + o1[1] * dist];
      const B1 = [p0[0] + o1[0] * dist, p0[1] + o1[1] * dist];
      const A2 = [p0[0] + o2[0] * dist, p0[1] + o2[1] * dist];
      const B2 = [pNext[0] + o2[0] * dist, pNext[1] + o2[1] * dist];
      const ip = islandLineIntersection(A1, B1, A2, B2);
      if (ip && isFinite(ip[0]) && isFinite(ip[1])) out.push(ip);
      else out.push([p0[0] + (o1[0] + o2[0]) * 0.5 * dist, p0[1] + (o1[1] + o2[1]) * 0.5 * dist]);
    }
    return out;
  }
  function islandFillRingEvenOdd(ctx, outerLoop, innerLoop) {
    if (!outerLoop || !innerLoop || outerLoop.length < 3 || innerLoop.length < 3) return;
    ctx.beginPath();
    ctx.moveTo(outerLoop[0][0], outerLoop[0][1]);
    for (let i = 1; i < outerLoop.length; i++) ctx.lineTo(outerLoop[i][0], outerLoop[i][1]);
    ctx.closePath();
    ctx.moveTo(innerLoop[0][0], innerLoop[0][1]);
    for (let j = 1; j < innerLoop.length; j++) ctx.lineTo(innerLoop[j][0], innerLoop[j][1]);
    ctx.closePath();
    ctx.fill('evenodd');
  }
  /**
   * Contour (kind='island'): single bright neon sky stroke; width ``widthM``
   * (metres, world). No pavement fills, inner ring, or tick marks.
   */
  function drawLayoutIslandMarkerLinesWorld(ctx, pts, sel, widthM) {
    const n = pts.length;
    if (n < 3) return;
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < n; i++) ctx.lineTo(pts[i][0], pts[i][1]);
    ctx.closePath();
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    const wM = Math.max(0.05, Number(widthM) || LAYOUT_ISLAND_WIDTH_DEFAULT_M);
    ctx.lineWidth = wM;
    ctx.strokeStyle = (!sel && layerMonoLinesOn()) ? c2dLayerMonoLineStrokeCss() : LAYOUT_ISLAND_STROKE_CSS;
    ctx.stroke();
    if (sel) {
      ctx.lineWidth = Math.max(wM * 0.6, 0.3);
      ctx.strokeStyle = '#38bdf8';
      ctx.stroke();
    }
  }
  function drawLayoutAreaMarkers2DFloor() {
    if (!ctx || !layoutMarkersVisible()) return;
    if (!layerIslandAreaFillEffective()) return;
    const vb = layoutWorldViewportAabbWithBufferM(LAYOUT_RENDER_VIEWPORT_BUFFER_M);
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.translate(state.panX, state.panY);
    ctx.scale(state.scale, state.scale);
    const fillArea = layerMonoFillOn() ? c2dLayerMonoFillDarkAsphaltCss() : c2dCssColorLightenSteps(c2dRunwayStroke(), 3);
    (state.layoutMarkers || []).forEach(function(m) {
      if (!m || m.kind !== 'area') return;
      const pts = m.points;
      if (!pts || pts.length < 3) return;
      const poly = pts.map(function(p) { return [Number(p.x), Number(p.y)]; }).filter(function(P) { return isFinite(P[0]) && isFinite(P[1]); });
      if (poly.length < 3) return;
      const areaAabb = pointsWorldAabb(poly);
      if (areaAabb && !aabbIntersectsViewport(vb, areaAabb)) return;
      ctx.beginPath();
      ctx.moveTo(poly[0][0], poly[0][1]);
      for (let j = 1; j < poly.length; j++) ctx.lineTo(poly[j][0], poly[j][1]);
      ctx.closePath();
      ctx.fillStyle = fillArea;
      ctx.fill();
    });
    if (state.markerDrawing && getMarkerSubKindFromPanel() === 'area' && state.markerAreaDraft && state.markerAreaDraft.points && state.markerAreaDraft.points.length) {
      const list = state.markerAreaDraft.points;
      const hw = state.markerAreaHoverWorld;
      const ptsArr = list.map(function(p) { return [p.x, p.y]; });
      const hoverArr = (hw && hw.length >= 2) ? hw : null;
      strokeLayoutPathDraftPolyline(ctx, ptsArr, hoverArr);
      drawLayoutPathDraftVertexDots(ctx, ptsArr, hoverArr);
      if (list.length >= 3 && hw && hw.length >= 2) {
        const c0 = list[0];
        const closeR = CELL_SIZE * TERM_CLOSE_POLY_CF;
        const dx = hw[0] - c0.x, dy = hw[1] - c0.y;
        if (dx * dx + dy * dy <= closeR * closeR) strokeLayoutPathDraftCloseHintArc(ctx, c0.x, c0.y, closeR);
      }
    }
    ctx.restore();
  }
  function drawLayoutIslandMarkers2DEarly() {
    if (!ctx || !layoutMarkersVisible()) return;
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.translate(state.panX, state.panY);
    ctx.scale(state.scale, state.scale);
    if (state.markerDrawing && getMarkerSubKindFromPanel() === 'island' && state.markerIslandDraft && state.markerIslandDraft.points && state.markerIslandDraft.points.length) {
      const list = state.markerIslandDraft.points;
      const hw = state.markerIslandHoverWorld;
      const ptsArr = list.map(function(p) { return [p.x, p.y]; });
      const hoverArr = (hw && hw.length >= 2) ? hw : null;
      strokeLayoutPathDraftPolyline(ctx, ptsArr, hoverArr);
      drawLayoutPathDraftVertexDots(ctx, ptsArr, hoverArr);
      if (list.length >= 3 && hw && hw.length >= 2) {
        const c0 = list[0];
        const closeR = CELL_SIZE * TERM_CLOSE_POLY_CF;
        const dx = hw[0] - c0.x, dy = hw[1] - c0.y;
        if (dx * dx + dy * dy <= closeR * closeR) strokeLayoutPathDraftCloseHintArc(ctx, c0.x, c0.y, closeR);
      }
    }
    ctx.restore();
  }
  function drawLayoutIslandMarkersOverlay2D() {
    if (!ctx || !layoutMarkersVisible()) return;
    const vb = layoutWorldViewportAabbWithBufferM(LAYOUT_RENDER_VIEWPORT_BUFFER_M);
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.translate(state.panX, state.panY);
    ctx.scale(state.scale, state.scale);
    const sel = state.selectedObject;
    if (layerIslandContourLinesEffective()) {
      (state.layoutMarkers || []).forEach(function(m) {
        if (!m || m.kind !== 'island') return;
        const pts = layoutIslandWorldPointsForDraw(m);
        if (pts.length < 3) return;
        const islandAabb = pointsWorldAabb(pts);
        if (islandAabb && !aabbIntersectsViewport(vb, islandAabb)) return;
        const isSel = !!(sel && sel.type === 'layoutMarker' && String(sel.id) === String(m.id));
        drawLayoutIslandMarkerLinesWorld(ctx, pts, isSel, islandWidthMResolved(m));
      });
    }
    ctx.restore();
  }
  function drawLayoutAreaMarkerOutlines2D() {
    if (!ctx || !layoutMarkersVisible()) return;
    const vb = layoutWorldViewportAabbWithBufferM(LAYOUT_RENDER_VIEWPORT_BUFFER_M);
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.translate(state.panX, state.panY);
    ctx.scale(state.scale, state.scale);
    const lwYellow = 0.5 * layoutHairlineStrokeWidthWorld();
    const sel = state.selectedObject;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    (state.layoutMarkers || []).forEach(function(m) {
      if (!m || m.kind !== 'area') return;
      const pts = m.points;
      if (!pts || pts.length < 3) return;
      const poly = pts.map(function(p) { return [Number(p.x), Number(p.y)]; }).filter(function(P) { return isFinite(P[0]) && isFinite(P[1]); });
      if (poly.length < 3) return;
      const areaAabb = pointsWorldAabb(poly);
      if (areaAabb && !aabbIntersectsViewport(vb, areaAabb)) return;
      const isSel = !!(sel && sel.type === 'layoutMarker' && String(sel.id) === String(m.id));
      ctx.beginPath();
      ctx.moveTo(poly[0][0], poly[0][1]);
      for (let j = 1; j < poly.length; j++) ctx.lineTo(poly[j][0], poly[j][1]);
      ctx.closePath();
      if (isSel) {
        ctx.fillStyle = c2dObjectSelectedFill();
        ctx.fill();
        ctx.lineWidth = Math.max(2.4 / Math.max(state.scale, 0.08), 3.2 * layoutHairlineStrokeWidthWorld());
        ctx.strokeStyle = c2dObjectSelectedStroke();
        ctx.shadowColor = c2dObjectSelectedGlow();
        ctx.shadowBlur = c2dObjectSelectedGlowBlur();
        ctx.stroke();
        ctx.shadowBlur = 0;
      }
      if (layerIslandContourLinesEffective() || isSel) {
        ctx.lineWidth = lwYellow;
        ctx.strokeStyle = (!isSel && layerMonoLinesOn()) ? c2dLayerMonoLineStrokeCss() : '#fcd410';
        ctx.stroke();
      }
    });
    ctx.restore();
  }
  function drawLayoutMarkers2D(interactiveLite) {
    if (!ctx || !layoutMarkersVisible()) return;
    const vb = layoutWorldViewportAabbWithBufferM(LAYOUT_RENDER_VIEWPORT_BUFFER_M);
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.translate(state.panX, state.panY);
    ctx.scale(state.scale, state.scale);
    const silN = Number(_acSil.noseX), silWR = Number(_acSil.wingRearX), silUY = Number(_acSil.wingUpperY);
    const silTN = Number(_acSil.tailNeckX), silLY = Number(_acSil.wingLowerY);
    const nX = isFinite(silN) ? silN : 0.6;
    const wRx = isFinite(silWR) ? silWR : -0.5;
    const uY = isFinite(silUY) ? silUY : 0.35;
    const tX = isFinite(silTN) ? silTN : -0.3;
    const lY = isFinite(silLY) ? silLY : -0.35;
    const planeScale = CELL_SIZE * 2.2;
    const markerTool = settingModeSelect && settingModeSelect.value === 'marker';
    if (state.markerTextDraft && state.markerTextDraft.active && (markerTool || state.layers.textRuler)) {
      const ax = state.markerTextDraft.x, ay = state.markerTextDraft.y;
      if (isFinite(ax) && isFinite(ay)) {
        ctx.beginPath();
        const rad = 5 / Math.max(state.scale, 0.08);
        ctx.arc(ax, ay, rad, 0, Math.PI * 2);
        ctx.fillStyle = '#38bdf8';
        ctx.fill();
        ctx.strokeStyle = 'rgba(255,255,255,0.92)';
        ctx.lineWidth = 1.25 / Math.max(state.scale, 0.08);
        ctx.stroke();
      }
    }
    (state.layoutMarkers || []).forEach(function(m) {
      if (!m) return;
      if (!markerTool) {
        if ((m.kind === 'text' || m.kind === 'ruler' || m.kind === 'navaid') && !state.layers.textRuler) return;
        if (m.kind === 'flight' && !state.layers.dummyFlight) return;
      }
      if (!(state.selectedObject && state.selectedObject.type === 'layoutMarker' && state.selectedObject.id === m.id)) {
        const mkAabb = markerWorldAabb(m);
        if (mkAabb && !aabbIntersectsViewport(vb, mkAabb)) return;
      }
      const sel = state.selectedObject && state.selectedObject.type === 'layoutMarker' && state.selectedObject.id === m.id;
      const etcM = layerMonoEtcOn();
      if (m.kind === 'text') {
        if (interactiveLite) return;
        const x = Number(m.x), y = Number(m.y);
        if (!isFinite(x) || !isFinite(y)) return;
        const fs = Math.max(10, 12 / Math.max(state.scale, 0.12));
        ctx.font = '600 ' + fs + 'px system-ui,sans-serif';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        const txt = String(m.text || '');
        ctx.lineWidth = Math.max(2, 3 / Math.max(state.scale, 0.1));
        if (etcM && !sel) {
          ctx.strokeStyle = 'rgba(30,41,59,0.55)';
          ctx.fillStyle = C2D_LAYER_MONO_ETC_WHITE;
        } else {
          ctx.strokeStyle = 'rgba(15,23,42,0.92)';
          ctx.fillStyle = '#e2e8f0';
        }
        ctx.strokeText(txt, x + 2, y + 2);
        ctx.fillText(txt, x + 2, y + 2);
        if (sel) layoutMarkerDrawEndpointDot(ctx, x, y, true);
        if (sel) {
          ctx.strokeStyle = '#38bdf8';
          ctx.lineWidth = 2 / Math.max(state.scale, 0.1);
          const tw = ctx.measureText(txt).width;
          ctx.strokeRect(x + 1, y + 1, tw + 4, fs + 4);
        }
      } else if (m.kind === 'ruler') {
        const x1 = Number(m.x1), y1 = Number(m.y1), x2 = Number(m.x2), y2 = Number(m.y2);
        if (![x1, y1, x2, y2].every(isFinite)) return;
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.strokeStyle = sel ? '#38bdf8' : (etcM ? C2D_LAYER_MONO_ETC_WHITE : '#94a3b8');
        ctx.lineWidth = Math.max(1.2, 1.6 / Math.max(state.scale, 0.1));
        ctx.setLineDash([6, 4]);
        ctx.stroke();
        ctx.setLineDash([]);
        const dx = x2 - x1, dy = y2 - y1;
        const lenM = Math.hypot(dx, dy);
        const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
        if (!interactiveLite) {
          const fs = Math.max(9, 10 / Math.max(state.scale, 0.12));
          ctx.font = '600 ' + fs + 'px system-ui,sans-serif';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          const label = lenM.toFixed(1) + ' m';
          ctx.lineWidth = 2.5;
          if (etcM && !sel) {
            ctx.strokeStyle = 'rgba(30,41,59,0.55)';
            ctx.fillStyle = C2D_LAYER_MONO_ETC_WHITE;
          } else {
            ctx.strokeStyle = 'rgba(15,23,42,0.85)';
            ctx.fillStyle = '#f1f5f9';
          }
          ctx.strokeText(label, mx, my - fs * 0.9);
          ctx.fillText(label, mx, my - fs * 0.9);
        }
        if (sel) {
          layoutMarkerDrawEndpointDot(ctx, x1, y1, true);
          layoutMarkerDrawEndpointDot(ctx, x2, y2, true);
        }
      } else if (m.kind === 'flight') {
        const pose = resolveMarkerFlightPose(m);
        if (!pose) return;
        ensureMarkerFlightBlazerState(m);
        if (m.blazerEnabled) {
          const lt = m.blazerLeftTrail || [];
          const rt = m.blazerRightTrail || [];
          function drawTrailBand(trail, oppositeTrail, colorHex) {
            if (!Array.isArray(trail) || trail.length < 2) return;
            const outer = [];
            const expandM = 5;
            for (let ti = 0; ti < trail.length; ti++) {
              const p = trail[ti];
              const op = (Array.isArray(oppositeTrail) && oppositeTrail.length > ti) ? oppositeTrail[ti] : null;
              let nx = 0;
              let ny = 0;
              if (op) {
                nx = Number(p.x) - Number(op.x);
                ny = Number(p.y) - Number(op.y);
              }
              const nl = Math.hypot(nx, ny) || 1;
              nx /= nl;
              ny /= nl;
              outer.push({ x: Number(p.x) + nx * expandM, y: Number(p.y) + ny * expandM });
            }
            ctx.save();
            ctx.globalAlpha = 0.4;
            ctx.beginPath();
            ctx.moveTo(Number(trail[0].x), Number(trail[0].y));
            for (let ti = 1; ti < trail.length; ti++) ctx.lineTo(Number(trail[ti].x), Number(trail[ti].y));
            for (let ti = outer.length - 1; ti >= 0; ti--) ctx.lineTo(Number(outer[ti].x), Number(outer[ti].y));
            ctx.closePath();
            ctx.fillStyle = colorHex;
            ctx.strokeStyle = colorHex;
            ctx.lineWidth = Math.max(0.45, 0.7 / Math.max(state.scale, 0.1));
            ctx.lineCap = 'round';
            ctx.lineJoin = 'round';
            ctx.fill();
            ctx.stroke();
            ctx.restore();
          }
          const trailColor = (etcM && !sel) ? C2D_LAYER_MONO_ETC_WHITE : (m.blazerColor || MARKER_BLAZER_COLOR_OPTIONS[0]);
          drawTrailBand(lt, rt, trailColor);
          drawTrailBand(rt, lt, trailColor);
        }
        const ac = getAircraftInfoByType(m.aircraftType);
        const lenM = ac && isFinite(Number(ac.length_m)) ? Math.max(1, Number(ac.length_m)) : 40;
        const spanM = ac && isFinite(Number(ac.wingspan_m)) ? Math.max(1, Number(ac.wingspan_m)) : 40;
        const useDetailSil = _ac2d.useDetailedSilhouette === true;
        const silhouette2D = getApronAircraftDetailedSilhouettePoints();
        ctx.save();
        ctx.translate(pose.x, pose.y);
        ctx.rotate(pose.ang);
        if (useDetailSil && silhouette2D.length >= 3) {
          const scaleX = lenM, scaleY = spanM;
          ctx.beginPath();
          ctx.moveTo(silhouette2D[0][0] * scaleX, silhouette2D[0][1] * scaleY);
          for (let si = 1; si < silhouette2D.length; si++) ctx.lineTo(silhouette2D[si][0] * scaleX, silhouette2D[si][1] * scaleY);
          ctx.closePath();
        } else {
          const scaleX = planeScale * 0.52, scaleY = planeScale * 0.38;
          ctx.beginPath();
          ctx.moveTo(scaleX * nX, 0);
          ctx.lineTo(scaleX * wRx, scaleY * uY);
          ctx.lineTo(scaleX * tX, 0);
          ctx.lineTo(scaleX * wRx, scaleY * lY);
          ctx.closePath();
        }
        ctx.fillStyle = sel ? c2dObjectSelectedFill() : (etcM ? C2D_LAYER_MONO_ETC_WHITE : '#94a3b8');
        ctx.strokeStyle = sel ? c2dObjectSelectedStroke() : (etcM ? C2D_LAYER_MONO_ETC_WHITE : 'rgba(30,41,59,0.9)');
        ctx.lineWidth = Math.max(0.75, 1.1 / Math.max(state.scale, 0.1));
        if (!sel) ctx.globalAlpha = 0.4;
        if (sel) {
          ctx.shadowColor = c2dObjectSelectedGlow();
          ctx.shadowBlur = c2dObjectSelectedGlowBlur();
        }
        ctx.fill();
        ctx.stroke();
        if (sel) layoutMarkerDrawEndpointDot(ctx, 0, 0, true);
        ctx.restore();
      } else if (m.kind === 'island' || m.kind === 'area') {
        const pts = m.kind === 'island' ? layoutIslandWorldPointsForDraw(m) : (m.points || []).map(function(p) {
          const x = Number(p.x), y = Number(p.y);
          return (isFinite(x) && isFinite(y)) ? [x, y] : null;
        }).filter(Boolean);
        if (pts.length < 3 || !sel) return;
        const sv = state.selectedVertex;
        for (let vi = 0; vi < pts.length; vi++) {
          const vSel = !!(sv && sv.type === 'layoutMarkerHandle' && sv.handle === 'islandVertex' && String(sv.id) === String(m.id) && sv.vertexIndex === vi);
          layoutMarkerDrawEndpointDot(ctx, pts[vi][0], pts[vi][1], vSel);
        }
      } else if (m.kind === 'navaid') {
        drawNavaidMarker2D(ctx, m, sel, interactiveLite);
      }
    });
    if (state.markerDrawing && state.markerRulerDraft && getMarkerSubKindFromPanel() === 'ruler' && state.markerRulerHoverWorld && (markerTool || state.layers.textRuler)) {
      const d0 = state.markerRulerDraft;
      const h = state.markerRulerHoverWorld;
      strokeLayoutPathDraftPolyline(ctx, [[d0.x, d0.y]], h);
      drawLayoutPathDraftVertexDots(ctx, [[d0.x, d0.y]], h);
      const dx = h[0] - d0.x, dy = h[1] - d0.y;
      const lenM = Math.hypot(dx, dy);
      const mx = (d0.x + h[0]) / 2, my = (d0.y + h[1]) / 2;
      if (!interactiveLite) {
        const fs = Math.max(9, 10 / Math.max(state.scale, 0.12));
        ctx.font = '600 ' + fs + 'px system-ui,sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        const label = lenM.toFixed(1) + ' m';
        ctx.lineWidth = 2.5;
        ctx.strokeStyle = 'rgba(15,23,42,0.85)';
        ctx.fillStyle = '#f1f5f9';
        ctx.strokeText(label, mx, my - fs * 0.9);
        ctx.fillText(label, mx, my - fs * 0.9);
      }
    }
    if (state.markerDrawing && getMarkerSubKindFromPanel() === 'flight' && state.markerFlightHoverSnap && (markerTool || state.layers.dummyFlight)) {
      const ghost = {
        kind: 'flight',
        taxiwayId: state.markerFlightHoverSnap.taxiwayId,
        segIndex: state.markerFlightHoverSnap.segIndex,
        t: state.markerFlightHoverSnap.t,
        aircraftType: getMarkerFlightAircraftTypeFromPanel(),
      };
      const pose = resolveMarkerFlightPose(ghost);
      if (pose) {
        const ac = getAircraftInfoByType(ghost.aircraftType);
        const lenM = ac && isFinite(Number(ac.length_m)) ? Math.max(1, Number(ac.length_m)) : 40;
        const spanM = ac && isFinite(Number(ac.wingspan_m)) ? Math.max(1, Number(ac.wingspan_m)) : 40;
        const useDetailSil = _ac2d.useDetailedSilhouette === true;
        const silhouette2D = getApronAircraftDetailedSilhouettePoints();
        ctx.save();
        ctx.translate(pose.x, pose.y);
        ctx.rotate(pose.ang);
        if (useDetailSil && silhouette2D.length >= 3) {
          ctx.beginPath();
          ctx.moveTo(silhouette2D[0][0] * lenM, silhouette2D[0][1] * spanM);
          for (let si = 1; si < silhouette2D.length; si++) ctx.lineTo(silhouette2D[si][0] * lenM, silhouette2D[si][1] * spanM);
          ctx.closePath();
        } else {
          const scaleX = planeScale * 0.52, scaleY = planeScale * 0.38;
          ctx.beginPath();
          ctx.moveTo(scaleX * nX, 0);
          ctx.lineTo(scaleX * wRx, scaleY * uY);
          ctx.lineTo(scaleX * tX, 0);
          ctx.lineTo(scaleX * wRx, scaleY * lY);
          ctx.closePath();
        }
        ctx.globalAlpha = 0.42;
        ctx.fillStyle = '#94a3b8';
        ctx.strokeStyle = 'rgba(30,41,59,0.75)';
        ctx.lineWidth = Math.max(0.75, 1.1 / Math.max(state.scale, 0.1));
        ctx.fill();
        ctx.globalAlpha = 1;
        ctx.stroke();
        ctx.restore();
      }
    }
    ctx.restore();
  }
  function drawPathArcPreview() {
    const d = state.pathArcDrag;
    if (!d || !d.previewPx || d.previewPx.length < 2) return;
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.translate(state.panX, state.panY);
    ctx.scale(state.scale, state.scale);
    ctx.beginPath();
    ctx.moveTo(d.previewPx[0][0], d.previewPx[0][1]);
    for (let i = 1; i < d.previewPx.length; i++) ctx.lineTo(d.previewPx[i][0], d.previewPx[i][1]);
    ctx.strokeStyle = 'rgba(244, 63, 94, 0.92)';
    ctx.lineWidth = Math.max(2, 2.2 / Math.max(state.scale, 0.08));
    ctx.setLineDash([6, 5]);
    ctx.lineCap = 'round';
    ctx.stroke();
    ctx.restore();
  }
  function draw(drawOpts) {
    if (!ctx || !layoutDrawCanvas) return;
    if (overlayCanvas && !overlayCtx) overlayCtx = overlayCanvas.getContext('2d');
    if (state.simSliderScrubbing && !(drawOpts && drawOpts.bypassSimScrubGuard)) return;
    const interactiveLite = layoutViewSkipsTaxiDetail(drawOpts);
    const simPlaybackSkipHeavyPathOverlays = !!(state.simPlaying && state.hasSimulationResult && !(drawOpts && drawOpts.forceFullLayoutDraw));
    const skipPathGeometryOverlays = !!(drawOpts && drawOpts.skipPathGeometryOverlays);
    function drawSimPlaybackBackgroundLayers() {
      drawGrid(interactiveLite);
      drawLayoutAreaMarkers2DFloor();
      drawLayoutAreaMarkerOutlines2D();
      drawLayoutIslandMarkers2DEarly();
      drawTerminals(interactiveLite);
      drawTaxiways(interactiveLite);
      drawLayoutIslandMarkersOverlay2D();
      drawPathArcPreview();
      drawHoldingPoints2D(interactiveLite);
    }
    const nowPerfDraw = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
    const viewRoughInteraction = !!(state.isPanning || nowPerfDraw < _layoutDetailSuppressUntil);
    /** 시뮬 재생 여부와 무관: 결과가 있고 뷰가 안정일 때 그리드~홀딩포인트까지 비트맵으로 재사용. 팬/휠 중에는 직접 그려 오프스크린 이중 작업 방지. */
    const layoutBgBitmapCache = !!(state.hasSimulationResult && !interactiveLite && !(drawOpts && drawOpts.forceFullLayoutDraw) && !viewRoughInteraction);
    if (layoutBgBitmapCache) {
      const w = layoutDrawCanvas.width, h = layoutDrawCanvas.height;
      ensureSimPlaybackBgCanvasBuffer(w, h);
      const sig = simPlaybackBackgroundCacheSignature(interactiveLite);
      if (sig !== _simPlaybackBgSig) {
        const ocan = _simPlaybackBgCanvas;
        const octx = ocan.getContext('2d', { alpha: false });
        const savedCtx = ctx;
        ctx = octx;
        octx.setTransform(1, 0, 0, 1, 0, 0);
        octx.clearRect(0, 0, w, h);
        try {
          drawSimPlaybackBackgroundLayers();
        } finally {
          ctx = savedCtx;
        }
        _simPlaybackBgSig = sig;
      }
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, w, h);
      ctx.drawImage(_simPlaybackBgCanvas, 0, 0);
    } else {
      drawSimPlaybackBackgroundLayers();
    }
    drawLayoutHeatmapOverlays();
    const wPxDraw = layoutDrawCanvas.width;
    const hPxDraw = layoutDrawCanvas.height;
    const useFg = layoutUseForegroundOverlay();
    if (useFg) {
      overlayCtx.setTransform(1, 0, 0, 1, 0, 0);
      overlayCtx.clearRect(0, 0, wPxDraw, hPxDraw);
    }
    const savedCtxDraw = ctx;
    if (useFg) ctx = overlayCtx;
    try {
      drawPBBs(interactiveLite);
      drawRemoteStands(interactiveLite);
      drawTempStands(interactiveLite);
      if (!interactiveLite || state.isPanning) drawApronTaxiwayLinks();
      drawStandPreview(interactiveLite);
      drawSelectedLayoutEdge();
      {
        const sel = state.selectedObject;
        const rid = state.flightPathRevealFlightId;
        if (sel && sel.type === 'flight' && rid != null && String(sel.id) === String(rid)) {
          drawFlightPathHighlight();
          drawDeparturePathHighlight();
        }
      }
      if (!simPlaybackSkipHeavyPathOverlays && !skipPathGeometryOverlays) drawProSimFlightPathEdges();
      drawHoldingQueueGhostFlights2D();
      drawFlights2D();
      if (!interactiveLite) {
        if (!simPlaybackSkipHeavyPathOverlays && !skipPathGeometryOverlays) drawPathJunctions();
        if (!skipPathGeometryOverlays) {
          drawTaxiwayDanglingEndpointMarks();
          drawQueueTaxiwayLaneMarkers();
        }
      }
      drawLayoutMarkers2D(interactiveLite);
    } finally {
      ctx = savedCtxDraw;
    }
    syncMarkerTextDraftInputPosition();
    syncMarkerFlightBlazerOverlayButton();
    updatePathArcHud();
  }

  function exportLayoutGroundRectSnapshotFor3D(x0, y0, widthM, heightM, meta) {
    if (!canvas || !ctx) return null;
    if (!(widthM > 0 && heightM > 0)) return null;
    const skipScheduleDraw = !!(meta && meta.skipScheduleDraw);
    const col = meta && meta.col != null ? meta.col : 0;
    const row = meta && meta.row != null ? meta.row : 0;
    const maxSidePx = 16384;
    const ppm = Math.max(0.35, Math.min(32, maxSidePx / Math.max(widthM, heightM)));
    let exportDpr = Math.min(5, Math.max(2, window.devicePixelRatio || 1));
    const logicalW = widthM * ppm;
    const logicalH = heightM * ppm;
    const maxExportCanvasDim = 16384;
    let wPx = Math.max(1, Math.round(logicalW * exportDpr));
    let hPx = Math.max(1, Math.round(logicalH * exportDpr));
    if (wPx > maxExportCanvasDim || hPx > maxExportCanvasDim) {
      const shrink = Math.min(maxExportCanvasDim / wPx, maxExportCanvasDim / hPx);
      exportDpr *= shrink;
      wPx = Math.max(1, Math.round(logicalW * exportDpr));
      hPx = Math.max(1, Math.round(logicalH * exportDpr));
    }
    const oc = document.createElement('canvas');
    oc.width = wPx;
    oc.height = hPx;
    const octx = oc.getContext('2d', { alpha: false });
    if (!octx) return null;
    if (typeof octx.imageSmoothingQuality === 'string') octx.imageSmoothingQuality = 'high';
    const savedLayoutDrawCanvas = layoutDrawCanvas;
    const savedCtx = ctx;
    const savedDpr = dpr;
    const savedPanX = state.panX;
    const savedPanY = state.panY;
    const savedScale = state.scale;
    const savedHoverCell = state.hoverCell;
    const savedSel = state.selectedObject;
    const savedVtx = state.selectedVertex;
    layoutDrawCanvas = oc;
    ctx = octx;
    dpr = exportDpr;
    state.panX = -x0 * ppm;
    state.panY = -y0 * ppm;
    state.scale = ppm;
    state.hoverCell = null;
    state.selectedObject = null;
    state.selectedVertex = null;
    invalidateGridUnderlay();
    try {
      draw({ bypassSimScrubGuard: true, forceFullLayoutDraw: true });
    } finally {
      layoutDrawCanvas = savedLayoutDrawCanvas;
      ctx = savedCtx;
      dpr = savedDpr;
      state.panX = savedPanX;
      state.panY = savedPanY;
      state.scale = savedScale;
      state.hoverCell = savedHoverCell;
      state.selectedObject = savedSel;
      state.selectedVertex = savedVtx;
      invalidateGridUnderlay();
      if (!skipScheduleDraw) scheduleDraw();
    }
    let dataUrl = '';
    try {
      dataUrl = oc.toDataURL('image/jpeg', 0.93);
    } catch (eJ) {
      try {
        dataUrl = oc.toDataURL('image/png');
      } catch (eP) {
        return null;
      }
    }
    if (!dataUrl || dataUrl.length < 48) return null;
    return {
      col: col,
      row: row,
      x0: x0,
      y0: y0,
      widthM: widthM,
      heightM: heightM,
      dataUrl: dataUrl,
      ppm: ppm,
      rasterWidthPx: wPx,
      rasterHeightPx: hPx
    };
  }

  function exportLayoutGroundTilesFor3D() {
    const maxWX = GRID_COLS * CELL_SIZE;
    const maxWY = GRID_ROWS * CELL_SIZE;
    if (!(maxWX > 0 && maxWY > 0)) return null;
    const mx = maxWX * 0.5;
    const my = maxWY * 0.5;
    const specs = [
      { col: 0, row: 0, x0: 0, y0: 0, widthM: mx, heightM: my },
      { col: 1, row: 0, x0: mx, y0: 0, widthM: maxWX - mx, heightM: my },
      { col: 0, row: 1, x0: 0, y0: my, widthM: mx, heightM: maxWY - my },
      { col: 1, row: 1, x0: mx, y0: my, widthM: maxWX - mx, heightM: maxWY - my }
    ];
    const tiles = [];
    for (let si = 0; si < specs.length; si++) {
      const sp = specs[si];
      const t = exportLayoutGroundRectSnapshotFor3D(sp.x0, sp.y0, sp.widthM, sp.heightM, { col: sp.col, row: sp.row, skipScheduleDraw: true });
      if (!t) {
        scheduleDraw();
        return null;
      }
      tiles.push(t);
    }
    scheduleDraw();
    return {
      version: 1,
      tileCols: 2,
      tileRows: 2,
      widthWorldM: maxWX,
      heightWorldM: maxWY,
      tiles: tiles
    };
  }

  function exportLayoutGroundTextureFor3D() {
    const maxWX = GRID_COLS * CELL_SIZE;
    const maxWY = GRID_ROWS * CELL_SIZE;
    if (!(maxWX > 0 && maxWY > 0)) return null;
    const t = exportLayoutGroundRectSnapshotFor3D(0, 0, maxWX, maxWY, { col: 0, row: 0, skipScheduleDraw: false });
    if (!t) return null;
    return {
      dataUrl: t.dataUrl,
      format: t.dataUrl.indexOf('jpeg') >= 0 ? 'image/jpeg' : 'image/png',
      widthWorldM: maxWX,
      heightWorldM: maxWY,
      pixelsPerMeter: t.ppm,
      rasterWidthPx: t.rasterWidthPx,
      rasterHeightPx: t.rasterHeightPx
    };
  }

  document.addEventListener('keydown', function(ev) {
    const el = document.activeElement;
    const inInput = el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.isContentEditable);
    if (ev.ctrlKey && ev.key === 'z') {
      if (!inInput) {
        ev.preventDefault();
        if (state.pathArcDrag) {
          state.pathArcDrag = null;
          updatePathArcHud();
          draw();
          return;
        }
        undo();
      }
      return;
    }
    if (ev.key === 'Escape') {
      if (!inInput && state.pathArcDrag) {
        ev.preventDefault();
        state.pathArcDrag = null;
        updatePathArcHud();
        draw();
        return;
      }
      if (state.markerTextDraft && state.markerTextDraft.active) {
        const inp = document.getElementById('markerTextDraftInput');
        if (document.activeElement === inp || !inInput) {
          ev.preventDefault();
          cancelMarkerTextDraftWithoutCommit();
          return;
        }
      }
      if (inInput) return;
      const anyLayoutDraw = !!(state.pbbDrawing || state.remoteDrawing || state.tempStandDrawing || state.holdingPointDrawing || state.apronLinkDrawing ||
        state.terminalDrawingId || state.taxiwayDrawingId || state.markerDrawing);
      if (anyLayoutDraw) {
        ev.preventDefault();
        cancelActiveLayoutDrawingState();
        state.terminalDrawingId = null;
        state.taxiwayDrawingId = null;
        syncPanelFromState();
        updateObjectInfo();
        if (typeof redrawLayoutAfterEdit === 'function') redrawLayoutAfterEdit();
        else if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths(); else draw();
        if (typeof syncAllocGanttSelectionHighlight === 'function') syncAllocGanttSelectionHighlight();
        return;
      }
      if (state.selectedObject || state.selectedVertex || state.pathArcModeOn) {
        ev.preventDefault();
        if (state.pathArcDrag) {
          state.pathArcDrag = null;
        }
        const soEsc = state.selectedObject;
        if (soEsc && soEsc.type === 'terminal' && state.currentTerminalId === soEsc.id) {
          state.currentTerminalId = state.terminals.length ? state.terminals[0].id : null;
        }
        state.pathArcModeOn = false;
        state.selectedObject = null;
        state.selectedVertex = null;
        state.flightPathRevealFlightId = null;
        syncPanelFromState();
        updateObjectInfo();
        updatePathArcHud();
        draw();
        if (typeof syncAllocGanttSelectionHighlight === 'function') syncAllocGanttSelectionHighlight();
        return;
      }
      return;
    }
    if (ev.key !== 'Delete' && ev.key !== 'Backspace') return;
    if (inInput) return;
    if (removeLastDrawingVertex()) {
      ev.preventDefault();
      return;
    }
    if (removeSelectedVertex()) {
      ev.preventDefault();
      return;
    }
    if (!state.selectedObject) return;
    const type = state.selectedObject.type;
    const id = state.selectedObject.id;
    if (type === 'layoutEdge') {
      state.selectedObject = null;
      state.selectedVertex = null;
      syncPanelFromState();
      updateObjectInfo();
      draw();
      ev.preventDefault();
      return;
    }
    if (type !== 'terminal' && type !== 'pbb' && type !== 'remote' && type !== 'tempStand' && type !== 'holdingPoint' && type !== 'taxiway' && type !== 'apronLink' && type !== 'flight' && type !== 'layoutMarker') return;
    pushUndo();
    removeLayoutObjectFromState(type, id);
    state.selectedObject = null;
    state.selectedVertex = null;
    if (type === 'terminal' && state.currentTerminalId === id) {
      state.currentTerminalId = state.terminals.length ? state.terminals[0].id : null;
      if (state.terminalDrawingId === id) {
        state.terminalDrawingId = null;
        state.layoutPathDrawPointer = null;
      }
    }
    if (type === 'taxiway' && state.taxiwayDrawingId === id) {
      state.taxiwayDrawingId = null;
      state.layoutPathDrawPointer = null;
    }
    syncPanelFromState();
    updateObjectInfo();
    if (typeof redrawLayoutAfterEdit === 'function') redrawLayoutAfterEdit();
    else if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths(); else draw();
    ev.preventDefault();
  });

  container.addEventListener('mousedown', function(ev) {
    if (ev.button !== 0) return;
    focusCanvasForLayoutHotkeys();
    const rect = canvas.getBoundingClientRect();
    const sx = ev.clientX - rect.left, sy = ev.clientY - rect.top;
    const [wx, wy] = screenToWorld(sx, sy);
    const mode = settingModeSelect.value;
    const pathArcStartOnCanvas = canvas && ev.target === canvas;
    if (state.pathArcModeOn && !state.pathArcDrag && pathArcStartOnCanvas) {
      const eligArc = isPathArcHudVertexSelection();
      if (eligArc && eligArc.kind === 'taxiway' && eligArc.tw.pathType !== 'runway') {
        const twPa = eligArc.tw, idxPa = eligArc.idx;
        if (idxPa > 0 && idxPa < (twPa.vertices || []).length - 1) {
          state.pathArcDrag = {
            taxiwayId: twPa.id,
            vertexIndex: idxPa,
            lastShift: !!ev.shiftKey,
            previewPx: pathArcComputePreviewWorldPx(twPa, idxPa, wx, wy)
          };
          updatePathArcHud();
          ev.preventDefault();
          draw();
          return;
        }
      } else if (eligArc && eligArc.kind === 'island') {
        const mkPa = eligArc.mk, idxPa = eligArc.idx;
        const ptsI = mkPa.points;
        const nI = (ptsI && ptsI.length) || 0;
        if (nI >= 3 && idxPa >= 0 && idxPa < nI) {
          const prev = ptsI[(idxPa - 1 + nI) % nI], next = ptsI[(idxPa + 1) % nI];
          state.pathArcDrag = {
            islandMarkerId: mkPa.id,
            vertexIndex: idxPa,
            lastShift: !!ev.shiftKey,
            previewPx: pathArcComputePreviewWorldPxFromAB(Number(prev.x), Number(prev.y), Number(next.x), Number(next.y), wx, wy)
          };
          updatePathArcHud();
          ev.preventDefault();
          draw();
          return;
        }
      } else if (eligArc && eligArc.kind === 'apronLink') {
        const lkA = eligArc.lk, viA = eligArc.polyVertexIndex;
        const polyA = getApronLinkPolylineWorldPts(lkA);
        if (polyA.length >= 3 && viA > 0 && viA < polyA.length - 1) {
          const Apx = polyA[viA - 1], Bpx = polyA[viA + 1];
          state.pathArcDrag = {
            apronLinkId: lkA.id,
            polyVertexIndex: viA,
            lastShift: !!ev.shiftKey,
            previewPx: pathArcComputePreviewWorldPxFromAB(Apx[0], Apx[1], Bpx[0], Bpx[1], wx, wy)
          };
          updatePathArcHud();
          ev.preventDefault();
          draw();
          return;
        }
      }
    }
    if (mode === 'terminal' && !state.terminalDrawingId) {
      const vhit = hitTestTerminalVertex(wx, wy);
      if (vhit) {
        pushUndo();
        state.dragVertex = vhit;
        state.selectedVertex = { type: 'terminal', id: vhit.terminalId, index: vhit.index };
        const term = state.terminals.find(t => t.id === vhit.terminalId);
        if (term) {
          state.flightPathRevealFlightId = null;
          state.selectedObject = { type: 'terminal', id: term.id, obj: term };
          state.currentTerminalId = term.id;
          syncPanelFromState();
          updateObjectInfo();
          draw();
        }
        return;
      }
    }
    if (state.selectedObject && state.selectedObject.type === 'taxiway') {
      const thit = hitTestTaxiwayVertex(wx, wy);
      if (thit && thit.taxiwayId === state.selectedObject.id) {
        pushUndo();
        state.dragTaxiwayVertex = thit;
        state.selectedVertex = { type: 'taxiway', id: thit.taxiwayId, index: thit.index };
        draw();
        return;
      }
    }
    const standRotateHit = hitTestStandRotationHandle(wx, wy);
    if (standRotateHit) {
      pushUndo();
      state.dragStandRotation = standRotateHit;
      state.selectedVertex = { type: 'standRotation', id: standRotateHit.id, standType: standRotateHit.type };
      draw();
      return;
    }
    if (state.selectedObject && state.selectedObject.type === 'pbb' && !state.pbbDrawing) {
      const ph = hitTestPbbEditablePoint(wx, wy);
      if (ph) {
        pushUndo();
        if (ph.type === 'bridge') {
          state.dragPbbBridgeVertex = { pbbId: state.selectedObject.id, bridgeIndex: ph.bridgeIndex, pointIndex: ph.pointIndex };
          state.selectedVertex = { type: 'pbbBridge', id: state.selectedObject.id, bridgeIndex: ph.bridgeIndex, pointIndex: ph.pointIndex };
        } else {
          state.dragStandConnection = { pbbId: state.selectedObject.id };
          state.selectedVertex = { type: 'pbbApronSite', id: state.selectedObject.id };
        }
        draw();
        return;
      }
    }
    if (state.selectedObject && state.selectedObject.type === 'remote' && !state.remoteDrawing) {
      const rh = hitTestRemoteStandDragPoint(wx, wy);
      if (rh) {
        pushUndo();
        state.dragRemoteStandPosition = { standId: state.selectedObject.id };
        state.selectedVertex = { type: 'remoteStandCenter', id: state.selectedObject.id };
        draw();
        return;
      }
    }
    if (state.selectedObject && state.selectedObject.type === 'apronLink' && !state.apronLinkDrawing) {
      const ah = hitTestApronLinkVertex(wx, wy);
      if (ah && ah.linkId === state.selectedObject.id) {
        pushUndo();
        state.dragApronLinkVertex = ah;
        state.selectedVertex = ah.kind === 'mid'
          ? { type: 'apronLink', id: ah.linkId, kind: 'mid', midIndex: ah.midIndex }
          : { type: 'apronLink', id: ah.linkId, kind: 'taxiway' };
        draw();
        return;
      }
    }
    if (layoutMarkersVisible() && state.selectedObject && state.selectedObject.type === 'layoutMarker' && !state.markerDrawing) {
      const mh = hitTestLayoutMarkerHandle(wx, wy);
      if (mh) {
        pushUndo();
        state.dragLayoutMarkerHandle = mh;
        state.selectedVertex = mh.handle === 'islandVertex'
          ? { type: 'layoutMarkerHandle', id: mh.markerId, handle: mh.handle, vertexIndex: mh.vertexIndex }
          : { type: 'layoutMarkerHandle', id: mh.markerId, handle: mh.handle };
        draw();
        return;
      }
    }
    state.selectedVertex = null;
    if ((mode === 'pbb' && state.pbbDrawing) || (mode === 'remote' && state.remoteDrawing) || (mode === 'tempStand' && state.tempStandDrawing) || (mode === 'holdingPoint' && state.holdingPointDrawing)) return;
    ev.preventDefault();
    state.dragStart = { sx, sy, panX: state.panX, panY: state.panY };
    state.isPanning = false;
    try {
      if (ev.pointerId != null && canvas && typeof canvas.setPointerCapture === 'function') {
        canvas.setPointerCapture(ev.pointerId);
      }
    } catch (e) {}
  });
  function clearCanvasPanGesture(ev) {
    if (!state.isPanning && !state.dragStart) return;
    state.dragStart = null;
    state.isPanning = false;
    try {
      if (ev && ev.pointerId != null && canvas && typeof canvas.releasePointerCapture === 'function') {
        canvas.releasePointerCapture(ev.pointerId);
      }
    } catch (e) {}
    flushDrawNow();
  }
  window.addEventListener('pointerup', function(ev) {
    if (ev.button !== 0) return;
    // Mouse selection is finalized in document mouseup; clearing here drops click-pick state.
    if (ev && String(ev.pointerType || '').toLowerCase() === 'mouse') return;
    clearCanvasPanGesture(ev);
  }, true);
  window.addEventListener('mouseup', function(ev) {
    if (ev.button !== 0) return;
    // Let document mouseup selection flow run first, then cleanup stale pan state.
    setTimeout(function() { clearCanvasPanGesture(); }, 0);
  }, true);
  function flushLayoutTooltipRaf() {
    if (_layoutTooltipRafId) {
      cancelAnimationFrame(_layoutTooltipRafId);
      _layoutTooltipRafId = 0;
    }
    _layoutTooltipPending = null;
  }
  function scheduleLayoutTooltipRaf(ev, wx, wy) {
    if (!flightTooltip || state.isPanning) return;
    _layoutTooltipPending = { ev: ev, wx: wx, wy: wy };
    if (_layoutTooltipRafId) return;
    _layoutTooltipRafId = requestAnimationFrame(function() {
      _layoutTooltipRafId = 0;
      const pack = _layoutTooltipPending;
      _layoutTooltipPending = null;
      if (!flightTooltip || !pack || !pack.ev) return;
      if (state.isPanning) {
        flightTooltip.style.display = 'none';
        return;
      }
      const ev2 = pack.ev, wxx = pack.wx, wyy = pack.wy;
      let tipDone = false;
      if (state.hasSimulationResult && state.globalUpdateFresh) {
        let bestFlight = null;
        let bestD2 = (CELL_SIZE * FLIGHT_TOOLTIP_CF) ** 2;
        const tSec = state.simTimeSec;
        state.flights.forEach(f => {
          const pose = getFlightPoseAtTimeForDraw(f, tSec);
          if (!pose || f.reg == null || !String(f.reg).trim()) return;
          const dx = pose.x - wxx;
          const dy = pose.y - wyy;
          const d2 = dx * dx + dy * dy;
          if (d2 < bestD2) { bestD2 = d2; bestFlight = f; }
        });
        if (bestFlight && bestFlight.reg) {
          flightTooltip.style.display = 'block';
          flightTooltip.textContent = String(bestFlight.reg).trim();
          flightTooltip.style.left = (ev2.clientX + 12) + 'px';
          flightTooltip.style.top = (ev2.clientY + 12) + 'px';
          tipDone = true;
        }
      }
      if (!tipDone) {
        const hit = hitTest(wxx, wyy);
        if (hit && hit.obj) {
          const name = (hit.obj.name != null && String(hit.obj.name).trim()) ? String(hit.obj.name).trim() : (hit.type === 'terminal' ? 'Building' : hit.type === 'pbb' ? 'Contact Stand' : hit.type === 'remote' ? 'Remote Stand' : hit.type === 'tempStand' ? 'Temp Stand' : hit.type === 'holdingPoint' ? holdingPointKindDisplayLabel(hit.obj.hpKind) : hit.type === 'taxiway' ? (hit.obj.name || 'Path') : hit.type === 'apronLink' ? (hit.obj.name || 'Apron Taxiway') : hit.type === 'layoutMarker' ? 'Marker' : hit.type);
          flightTooltip.style.display = 'block';
          flightTooltip.textContent = name;
          flightTooltip.style.left = (ev2.clientX + 12) + 'px';
          flightTooltip.style.top = (ev2.clientY + 12) + 'px';
        } else {
          flightTooltip.style.display = 'none';
        }
      }
    });
  }
  function layoutSnapDragActive() {
    return !!(state.pathArcDrag || state.dragVertex || state.dragTaxiwayVertex || state.dragPbbBridgeVertex || state.dragStandConnection || state.dragRemoteStandPosition || state.dragApronLinkVertex || state.dragStandRotation || state.dragLayoutMarkerHandle);
  }
  function onLayoutCanvasCoordinateMove(ev) {
    const rect = canvas.getBoundingClientRect();
    const sx = ev.clientX - rect.left, sy = ev.clientY - rect.top;
    const [wx, wy] = screenToWorld(sx, sy);
    const snappedPt = worldPointToCellPoint(wx, wy, !!ev.shiftKey);
    const snappedPx = cellToPixel(snappedPt.col, snappedPt.row);
    const [col, row] = pixelToCell(wx, wy);
    const cellKey = String(col) + ',' + String(row);
    if (cellKey !== _layoutReadoutLastCellKey) {
      _layoutReadoutLastCellKey = cellKey;
      if (coordEl) coordEl.textContent = 'cell: (' + col + ', ' + row + ')';
    }
    const pixelStr = 'x: ' + wx.toFixed(1) + '  y: ' + wy.toFixed(1);
    if (pixelStr !== _layoutReadoutLastPixelStr) {
      _layoutReadoutLastPixelStr = pixelStr;
      if (cursorPixelReadoutEl) cursorPixelReadoutEl.textContent = pixelStr;
    }
    const prev = state.hoverCell;
    state.hoverCell = { col, row };
    const hoverChanged = !prev || prev.col !== col || prev.row !== row;
    let drewThisMove = false;
    if (settingModeSelect.value === 'apronTaxiway' && state.apronLinkDrawing && state.apronLinkTemp) {
      const pw = state.apronLinkPointerWorld;
      if (!pw || pw[0] !== wx || pw[1] !== wy) {
        state.apronLinkPointerWorld = [wx, wy];
        scheduleDraw(); drewThisMove = true;
      }
    } else if (state.apronLinkPointerWorld) {
      state.apronLinkPointerWorld = null;
      scheduleDraw(); drewThisMove = true;
    }
    if (settingModeSelect.value === 'marker' && state.markerDrawing && getMarkerSubKindFromPanel() === 'ruler' && state.markerRulerDraft) {
      const hpx = worldPointToPixel(wx, wy, !!ev.shiftKey);
      const pw = state.markerRulerHoverWorld;
      if (!pw || pw[0] !== hpx[0] || pw[1] !== hpx[1]) {
        state.markerRulerHoverWorld = [hpx[0], hpx[1]];
        scheduleDraw(); drewThisMove = true;
      }
    } else if (state.markerRulerHoverWorld) {
      state.markerRulerHoverWorld = null;
      if (!drewThisMove) { scheduleDraw(); drewThisMove = true; }
    }
    if (settingModeSelect.value === 'marker' && state.markerDrawing && getMarkerSubKindFromPanel() === 'island' && state.markerIslandDraft) {
      const hpxI = worldPointToPixel(wx, wy, !!ev.shiftKey);
      const pwI = state.markerIslandHoverWorld;
      if (!pwI || pwI[0] !== hpxI[0] || pwI[1] !== hpxI[1]) {
        state.markerIslandHoverWorld = [hpxI[0], hpxI[1]];
        scheduleDraw(); drewThisMove = true;
      }
    } else if (state.markerIslandHoverWorld) {
      state.markerIslandHoverWorld = null;
      if (!drewThisMove) { scheduleDraw(); drewThisMove = true; }
    }
    if (settingModeSelect.value === 'marker' && state.markerDrawing && getMarkerSubKindFromPanel() === 'area' && state.markerAreaDraft) {
      const hpxA = markerAreaSnapWorldToPlacementPx(wx, wy, !!ev.shiftKey);
      const pwA = state.markerAreaHoverWorld;
      if (!pwA || pwA[0] !== hpxA[0] || pwA[1] !== hpxA[1]) {
        state.markerAreaHoverWorld = [hpxA[0], hpxA[1]];
        scheduleDraw(); drewThisMove = true;
      }
    } else if (state.markerAreaHoverWorld) {
      state.markerAreaHoverWorld = null;
      if (!drewThisMove) { scheduleDraw(); drewThisMove = true; }
    }
    if (settingModeSelect.value === 'marker' && state.markerDrawing && getMarkerSubKindFromPanel() === 'flight') {
      const snap = snapWorldToMarkerFlightTaxiway(wx, wy);
      const prev = state.markerFlightHoverSnap;
      if (snap) {
        const same = prev && prev.taxiwayId === snap.taxiwayId && prev.segIndex === snap.segIndex && Math.abs(prev.t - snap.t) < 1e-6;
        if (!same) {
          state.markerFlightHoverSnap = { taxiwayId: snap.taxiwayId, segIndex: snap.segIndex, t: snap.t };
          scheduleDraw(); drewThisMove = true;
        }
      } else if (prev) {
        state.markerFlightHoverSnap = null;
        if (!drewThisMove) { scheduleDraw(); drewThisMove = true; }
      }
    } else if (state.markerFlightHoverSnap) {
      state.markerFlightHoverSnap = null;
      if (!drewThisMove) { scheduleDraw(); drewThisMove = true; }
    }
    const pathLayoutDrawing = !!(state.terminalDrawingId || state.taxiwayDrawingId);
    const blockLayoutPathPtr = !!(state.isPanning || state.pathArcDrag || state.dragVertex || state.dragTaxiwayVertex || state.dragPbbBridgeVertex || state.dragStandConnection || state.dragRemoteStandPosition || state.dragApronLinkVertex || state.dragStandRotation || state.dragLayoutMarkerHandle);
    if (pathLayoutDrawing && !blockLayoutPathPtr) {
      const nx = snappedPx[0], ny = snappedPx[1];
      const lp = state.layoutPathDrawPointer;
      if (!lp || lp[0] !== nx || lp[1] !== ny) {
        state.layoutPathDrawPointer = [nx, ny];
        scheduleDraw(); drewThisMove = true;
      }
    } else if (state.layoutPathDrawPointer && (!pathLayoutDrawing || blockLayoutPathPtr)) {
      state.layoutPathDrawPointer = null;
      if (!drewThisMove) { scheduleDraw(); drewThisMove = true; }
    }
    if (state.pathArcDrag) {
      const d = state.pathArcDrag;
      if (d.islandMarkerId != null) {
        const mkArc = (state.layoutMarkers || []).find(function(m) { return m && String(m.id) === String(d.islandMarkerId); });
        const ptsA = mkArc && mkArc.points;
        const nA = (ptsA && ptsA.length) || 0;
        const vi = d.vertexIndex;
        if (!mkArc || !isLayoutPolygonMarkerKind(mkArc.kind) || nA < 3 || vi < 0 || vi >= nA) {
          state.pathArcDrag = null;
          updatePathArcHud();
          if (!drewThisMove) scheduleDraw();
        } else {
          const prev = ptsA[(vi - 1 + nA) % nA], next = ptsA[(vi + 1) % nA];
          d.lastShift = !!ev.shiftKey;
          d.previewPx = pathArcComputePreviewWorldPxFromAB(Number(prev.x), Number(prev.y), Number(next.x), Number(next.y), wx, wy);
          scheduleDraw();
          drewThisMove = true;
        }
      } else if (d.apronLinkId != null) {
        const lkArc = (state.apronLinks || []).find(function(l) { return l && l.id === d.apronLinkId; });
        const vi = d.polyVertexIndex;
        const polyA = lkArc ? getApronLinkPolylineWorldPts(lkArc) : [];
        if (!lkArc || vi <= 0 || vi >= polyA.length - 1) {
          state.pathArcDrag = null;
          updatePathArcHud();
          if (!drewThisMove) scheduleDraw();
        } else {
          const Apx = polyA[vi - 1], Bpx = polyA[vi + 1];
          d.lastShift = !!ev.shiftKey;
          d.previewPx = pathArcComputePreviewWorldPxFromAB(Apx[0], Apx[1], Bpx[0], Bpx[1], wx, wy);
          scheduleDraw();
          drewThisMove = true;
        }
      } else {
        const twArc = state.taxiways.find(function(t) { return t.id === d.taxiwayId; });
        if (!twArc || twArc.pathType === 'runway' || d.vertexIndex <= 0 || d.vertexIndex >= (twArc.vertices || []).length - 1) {
          state.pathArcDrag = null;
          updatePathArcHud();
          if (!drewThisMove) scheduleDraw();
        } else {
          d.lastShift = !!ev.shiftKey;
          d.previewPx = pathArcComputePreviewWorldPx(twArc, d.vertexIndex, wx, wy);
          scheduleDraw();
          drewThisMove = true;
        }
      }
      return;
    }
    if (state.dragVertex) {
      const term = state.terminals.find(t => t.id === state.dragVertex.terminalId);
      if (term && term.vertices[state.dragVertex.index]) {
        const v = term.vertices[state.dragVertex.index];
        if (v.col === snappedPt.col && v.row === snappedPt.row) return;
        v.col = snappedPt.col;
        v.row = snappedPt.row;
        scheduleDraw(); drewThisMove = true;
      }
      return;
    }
    if (state.dragTaxiwayVertex) {
      const tw = state.taxiways.find(t => t.id === state.dragTaxiwayVertex.taxiwayId);
      if (tw && tw.vertices[state.dragTaxiwayVertex.index]) {
        const v = tw.vertices[state.dragTaxiwayVertex.index];
        if (v.col === snappedPt.col && v.row === snappedPt.row) return;
        v.col = snappedPt.col;
        v.row = snappedPt.row;
        scheduleDraw(); drewThisMove = true;
      }
      return;
    }
    if (state.dragStandRotation) {
      if (state.dragStandRotation.type === 'pbb') {
        const pbb = state.pbbStands.find(function(item) { return item.id === state.dragStandRotation.id; });
        if (pbb) {
          const origin = getPbbRotationOriginPx(pbb);
          const nextDeg = normalizeAngleDeg(Math.atan2(wy - origin[1], wx - origin[0]) * 180 / Math.PI);
          if (Math.abs(nextDeg - (Number(pbb.angleDeg) || 0)) < 1e-4) return;
          pbb.angleDeg = nextDeg;
          const angleInput = document.getElementById('standAngle');
          if (angleInput) angleInput.value = String(Math.round(getPbbAngleDeg(pbb)));
          scheduleDraw(); drewThisMove = true;
        }
      } else if (state.dragStandRotation.type === 'remote') {
        const st = state.remoteStands.find(function(item) { return item.id === state.dragStandRotation.id; });
        if (st) {
          const center = getRemoteStandCenterPx(st);
          const nextDeg = normalizeAngleDeg(Math.atan2(wy - center[1], wx - center[0]) * 180 / Math.PI);
          if (Math.abs(nextDeg - (Number(st.angleDeg) || 0)) < 1e-4) return;
          st.angleDeg = nextDeg;
          scheduleDraw(); drewThisMove = true;
        }
      } else if (state.dragStandRotation.type === 'tempStand') {
        const st = (state.tempStands || []).find(function(item) { return item.id === state.dragStandRotation.id; });
        if (st) {
          const center = getRemoteStandCenterPx(st);
          const nextDeg = normalizeAngleDeg(Math.atan2(wy - center[1], wx - center[0]) * 180 / Math.PI);
          if (Math.abs(nextDeg - (Number(st.angleDeg) || 0)) < 1e-4) return;
          st.angleDeg = nextDeg;
          scheduleDraw(); drewThisMove = true;
        }
      }
      return;
    }
    if (state.dragPbbBridgeVertex) {
      const pbb = state.pbbStands.find(function(item) { return item.id === state.dragPbbBridgeVertex.pbbId; });
      if (pbb && Array.isArray(pbb.pbbBridges) && pbb.pbbBridges[state.dragPbbBridgeVertex.bridgeIndex] && Array.isArray(pbb.pbbBridges[state.dragPbbBridgeVertex.bridgeIndex].points) && pbb.pbbBridges[state.dragPbbBridgeVertex.bridgeIndex].points[state.dragPbbBridgeVertex.pointIndex]) {
        const pt = pbb.pbbBridges[state.dragPbbBridgeVertex.bridgeIndex].points[state.dragPbbBridgeVertex.pointIndex];
        if (state.dragPbbBridgeVertex.pointIndex === 0) {
          const proj = getClosestTerminalEdgePoint(wx, wy);
          if (proj && proj.point && proj.term) {
            const fr = getPbbTerminalFrameFromEdge(proj.term, proj.edgeIndex, proj.point[0], proj.point[1]);
            const wx = proj.point[0], wy = proj.point[1];
            const Tx = wx, Ty = wy;
            const bh = getPbbBoardingHeightM(pbb);
            const Bx = Tx + fr.nx * bh, By = Ty + fr.ny * bh;
            pbb.x1 = Tx;
            pbb.y1 = Ty;
            pbb.x2 = Bx;
            pbb.y2 = By;
            pbb.pbbBridges.forEach(function(bridge) {
              if (!bridge.points || bridge.points.length < 3) return;
              bridge.points[0].x = Tx;
              bridge.points[0].y = Ty;
              bridge.points[1].x = Bx;
              bridge.points[1].y = By;
            });
            bumpPathPolylineCacheRev();
          }
        } else {
          const nx = snappedPx[0], ny = snappedPx[1];
          if (Math.abs(pt.x - nx) < 1e-5 && Math.abs(pt.y - ny) < 1e-5) return;
          pt.x = nx;
          pt.y = ny;
        }
        scheduleDraw(); drewThisMove = true;
      }
      return;
    }
    if (state.dragStandConnection) {
      const pbb = state.pbbStands.find(function(item) { return item.id === state.dragStandConnection.pbbId; });
      if (pbb) {
        const nx = snappedPx[0], ny = snappedPx[1];
        const prev = getStandConnectionPx(pbb);
        if (Math.abs(prev[0] - nx) < 1e-5 && Math.abs(prev[1] - ny) < 1e-5) return;
        pbb.apronSiteX = nx;
        pbb.apronSiteY = ny;
        scheduleDraw(); drewThisMove = true;
      }
      return;
    }
    if (state.dragRemoteStandPosition) {
      const st = state.remoteStands.find(function(item) { return item.id === state.dragRemoteStandPosition.standId; });
      if (st) {
        if (st.col === snappedPt.col && st.row === snappedPt.row) return;
        st.x = snappedPx[0];
        st.y = snappedPx[1];
        st.col = snappedPt.col;
        st.row = snappedPt.row;
        scheduleDraw(); drewThisMove = true;
      }
      return;
    }
    if (state.dragApronLinkVertex) {
      const lk = state.apronLinks.find(l => l.id === state.dragApronLinkVertex.linkId);
      if (!lk) {
        state.dragApronLinkVertex = null;
      } else if (state.dragApronLinkVertex.kind === 'mid') {
        const mi = state.dragApronLinkVertex.midIndex;
        if (lk.midVertices && mi >= 0 && mi < lk.midVertices.length &&
            col >= 0 && row >= 0 && col <= GRID_COLS && row <= GRID_ROWS) {
          const mv = lk.midVertices[mi];
          if (mv.col === snappedPt.col && mv.row === snappedPt.row) return;
          mv.col = snappedPt.col;
          mv.row = snappedPt.row;
          markApronLinkJunctionOverlayDirty(lk.id);
          scheduleDraw(); drewThisMove = true;
        }
      } else if (state.dragApronLinkVertex.kind === 'taxiway') {
        const snap = snapWorldPointToTaxiwayPolyline(wx, wy, lk.taxiwayId);
        if (snap) {
          if (Math.abs(lk.tx - snap[0]) < 1e-5 && Math.abs(lk.ty - snap[1]) < 1e-5) return;
          lk.tx = snap[0];
          lk.ty = snap[1];
          markApronLinkJunctionOverlayDirty(lk.id);
          scheduleDraw(); drewThisMove = true;
        }
      }
      return;
    }
    if (state.dragLayoutMarkerHandle) {
      const h = state.dragLayoutMarkerHandle;
      const mk = (state.layoutMarkers || []).find(function(m) { return m && String(m.id) === String(h.markerId); });
      if (!mk) {
        state.dragLayoutMarkerHandle = null;
      } else {
        const px = worldPointToPixel(wx, wy, !!ev.shiftKey);
        if (h.handle === 'textAnchor') {
          mk.x = px[0];
          mk.y = px[1];
        } else if (h.handle === 'rulerA') {
          mk.x1 = px[0];
          mk.y1 = px[1];
        } else if (h.handle === 'rulerB') {
          mk.x2 = px[0];
          mk.y2 = px[1];
        } else if (h.handle === 'islandVertex') {
          const vi = h.vertexIndex;
          if (typeof vi === 'number' && mk.points && vi >= 0 && vi < mk.points.length) {
            const pxDrag = mk.kind === 'area' ? markerAreaSnapWorldToPlacementPx(wx, wy, !!ev.shiftKey) : worldPointToPixel(wx, wy, !!ev.shiftKey);
            mk.points[vi].x = pxDrag[0];
            mk.points[vi].y = pxDrag[1];
          }
        } else if (h.handle === 'flightCenter') {
          const prevPose = resolveMarkerFlightPose(mk);
          const snap = snapWorldToMarkerFlightTaxiway(wx, wy, { allowFar: true });
          if (snap) {
            const cand = { kind: 'flight', taxiwayId: snap.taxiwayId, segIndex: snap.segIndex, t: snap.t };
            const nextPose = resolveMarkerFlightPose(cand);
            const maxStep = Math.max(CELL_SIZE * 2.2, 22 / Math.max(state.scale, 0.1));
            const canMove = !prevPose || !nextPose || dist2([prevPose.x, prevPose.y], [nextPose.x, nextPose.y]) <= (maxStep * maxStep);
            if (canMove) {
              mk.taxiwayId = snap.taxiwayId;
              mk.segIndex = snap.segIndex;
              mk.t = snap.t;
              appendMarkerFlightBlazerTrail(mk);
            }
          }
        } else if (h.handle === 'navaidCenter') {
          mk.x = px[0];
          mk.y = px[1];
        }
        if (state.selectedObject && state.selectedObject.type === 'layoutMarker' && String(state.selectedObject.id) === String(h.markerId))
          state.selectedObject.obj = mk;
        syncMarkerTextDraftInputPosition();
        scheduleDraw(); drewThisMove = true;
      }
      return;
    }
    if (state.dragStart) {
      if ((ev.buttons & 1) === 0) {
        state.dragStart = null;
        state.isPanning = false;
      } else {
        const dx = sx - state.dragStart.sx, dy = sy - state.dragStart.sy;
        if (!state.isPanning && (Math.abs(dx) > DRAG_THRESH || Math.abs(dy) > DRAG_THRESH))
          state.isPanning = true;
        if (state.isPanning) {
          state.panX = state.dragStart.panX + dx;
          state.panY = state.dragStart.panY + dy;
          scheduleDraw(); drewThisMove = true;
        }
      }
    }
    const mode = settingModeSelect.value;
    if (!state.isPanning && !state.dragVertex && mode === 'holdingPoint' && state.holdingPointDrawing) {
      const snap = snapHoldingPointOnAllowedTaxiways(wx, wy);
      const nextHp = snap ? { x: snap.x, y: snap.y, pathType: snap.pathType } : null;
      const prevHp = state.previewHoldingPoint;
      const hpSame = (nextHp == null && prevHp == null) || (nextHp && prevHp && nextHp.x === prevHp.x && nextHp.y === prevHp.y && nextHp.pathType === prevHp.pathType);
      if (!hpSame) {
        state.previewHoldingPoint = nextHp;
        scheduleDraw(); drewThisMove = true;
      }
    } else if (!state.isPanning && !state.dragVertex && mode === 'remote' && state.remoteDrawing) {
      const category = panelRepresentativeCategoryForNewStand('remote');
      const angleDeg = 0;
      const candidate = { x: snappedPx[0], y: snappedPx[1], category, angleDeg };
      const candCorners = getRemoteStandCorners(candidate);
      let overlap = false;
      for (let i = 0; i < state.remoteStands.length; i++) {
        const o = state.remoteStands[i];
        if (standFootprintsTooClose(candCorners, category, getRemoteStandCorners(o), o.category || 'C')) { overlap = true; break; }
      }
      if (!overlap) {
        for (let i = 0; i < state.pbbStands.length; i++) {
          const o = state.pbbStands[i];
          if (standFootprintsTooClose(candCorners, category, getPBBStandCorners(o), o.category || 'C')) { overlap = true; break; }
        }
      }
      if (!overlap) {
        const tempsPv = state.tempStands || [];
        for (let i = 0; i < tempsPv.length; i++) {
          const o = tempsPv[i];
          if (standFootprintsTooClose(candCorners, category, getRemoteStandCorners(o), o.category || 'C')) { overlap = true; break; }
        }
      }
      if (!overlap && standGapLineHitsExistingOuterContours([candidate.x, candidate.y], angleDeg * Math.PI / 180, category)) overlap = true;
      const maxX = GRID_COLS * CELL_SIZE, maxY = GRID_ROWS * CELL_SIZE;
      if (candidate.x < 0 || candidate.y < 0 || candidate.x > maxX || candidate.y > maxY) overlap = true;
      const nextRem = { x: candidate.x, y: candidate.y, overlap };
      const prevRem = state.previewRemote;
      const remSame = prevRem && prevRem.x === nextRem.x && prevRem.y === nextRem.y && !!prevRem.overlap === !!nextRem.overlap;
      if (!remSame) {
        state.previewRemote = nextRem;
        scheduleDraw(); drewThisMove = true;
      }
    } else if (!state.isPanning && !state.dragVertex && mode === 'tempStand' && state.tempStandDrawing) {
      const snap = snapTempStandOnTaxiwayCenterlines(wx, wy);
      if (!snap) {
        if (state.previewTempStand != null) {
          state.previewTempStand = null;
          scheduleDraw(); drewThisMove = true;
        }
      } else {
        const category = panelRepresentativeCategoryForNewStand('tempStand');
        const angleDeg = 0;
        const candidate = { x: snap.x, y: snap.y, category, angleDeg };
        const candCorners = getRemoteStandCorners(candidate);
        let overlap = false;
        const temps = state.tempStands || [];
        for (let i = 0; i < temps.length; i++) {
          const o = temps[i];
          if (standFootprintsTooClose(candCorners, category, getRemoteStandCorners(o), o.category || 'C')) { overlap = true; break; }
        }
        if (!overlap) {
          for (let i = 0; i < state.remoteStands.length; i++) {
            const o = state.remoteStands[i];
            if (standFootprintsTooClose(candCorners, category, getRemoteStandCorners(o), o.category || 'C')) { overlap = true; break; }
          }
        }
        if (!overlap) {
          for (let i = 0; i < state.pbbStands.length; i++) {
            const o = state.pbbStands[i];
            if (standFootprintsTooClose(candCorners, category, getPBBStandCorners(o), o.category || 'C')) { overlap = true; break; }
          }
        }
        if (!overlap && standGapLineHitsExistingOuterContours([candidate.x, candidate.y], angleDeg * Math.PI / 180, category)) overlap = true;
        const maxX = GRID_COLS * CELL_SIZE, maxY = GRID_ROWS * CELL_SIZE;
        if (candidate.x < 0 || candidate.y < 0 || candidate.x > maxX || candidate.y > maxY) overlap = true;
        const nextTs = { x: candidate.x, y: candidate.y, overlap };
        const prevTs = state.previewTempStand;
        const tsSame = prevTs && prevTs.x === nextTs.x && prevTs.y === nextTs.y && !!prevTs.overlap === !!nextTs.overlap;
        if (!tsSame) {
          state.previewTempStand = nextTs;
          scheduleDraw(); drewThisMove = true;
        }
      }
    } else if (!state.isPanning && !state.dragVertex && mode === 'pbb' && state.pbbDrawing) {
      let bestEdge = null, bestD2 = Infinity;
      state.terminals.forEach(t => {
        if (!t.closed || t.vertices.length < 2) return;
        let tcx = 0, tcy = 0;
        t.vertices.forEach(v => { const q = cellToPixel(v.col, v.row); tcx += q[0]; tcy += q[1]; });
        tcx /= t.vertices.length || 1;
        tcy /= t.vertices.length || 1;
        for (let i = 0; i < t.vertices.length; i++) {
          const v1 = t.vertices[i], v2 = t.vertices[(i+1) % t.vertices.length];
          const p1 = cellToPixel(v1.col, v1.row), p2 = cellToPixel(v2.col, v2.row);
          const near = closestPointOnSegment(p1, p2, snappedPx);
          if (near) {
            const d2 = dist2(near, snappedPx);
            if (d2 < bestD2) { bestD2 = d2; bestEdge = { near: near, p1: p1, p2: p2, cx: tcx, cy: tcy }; }
          }
        }
      });
      const maxD2 = (CELL_SIZE * TRY_PBB_MAX_EDGE_CF) ** 2;
      if (bestEdge && bestD2 < maxD2) {
        const nearPt = bestEdge.near;
        const ex = (nearPt && nearPt[0] != null) ? nearPt[0] : 0;
        const ey = (nearPt && nearPt[1] != null) ? nearPt[1] : 0;
        const [x1,y1]=bestEdge.p1, [x2,y2]=bestEdge.p2;
        let nx = -(y2-y1), ny = x2-x1;
        const len = Math.hypot(nx,ny) || 1; nx /= len; ny /= len;
        const inX = bestEdge.cx - ex, inY = bestEdge.cy - ey;
        if (nx * inX + ny * inY > 0) { nx *= -1; ny *= -1; }
        const category = panelRepresentativeCategoryForNewStand('pbb');
        const bhEl = document.getElementById('pbbBoardingHeight');
        const previewBh = Math.max(0.5, Number(bhEl && bhEl.value) || 15);
        const wallX = ex, wallY = ey;
        const px2 = wallX + nx * previewBh, py2 = wallY + ny * previewBh;
        const cfgRow = standConfigRowForIcaoCat(category);
        const noseClear = cfgRow ? Number(cfgRow.nose_clear) : NaN;
        const offM = (Number.isFinite(noseClear) && noseClear > 0)
          ? noseClear
          : PBB_STAND_CENTER_OFFSET_FROM_TERMINAL_WALL_M;
        const previewAng = normalizeAngleDeg(Math.atan2(ny, nx) * 180 / Math.PI);
        const preview = {
          x1: wallX, y1: wallY, x2: px2, y2: py2, category,
          angleDeg: previewAng,
          apronSiteX: wallX + nx * offM,
          apronSiteY: wallY + ny * offM,
          terminalContactSetbackM: offM
        };
        const overlap = pbbStandOverlapsExisting(preview);
        const warnOuterOverlap = !overlap && pbbStandOuterContoursOverlapExisting(preview);
        const nextPbb = {
          x1: wallX, y1: wallY, x2: px2, y2: py2, category: preview.category,
          overlap, warnOuterOverlap, angleDeg: previewAng, apronSiteX: preview.apronSiteX, apronSiteY: preview.apronSiteY
        };
        const prevPbb = state.previewPbb;
        const pbbSame = prevPbb && prevPbb.x1 === nextPbb.x1 && prevPbb.y1 === nextPbb.y1 && prevPbb.x2 === nextPbb.x2 && prevPbb.y2 === nextPbb.y2 && String(prevPbb.category || '') === String(nextPbb.category || '') && !!prevPbb.overlap === !!nextPbb.overlap && !!prevPbb.warnOuterOverlap === !!nextPbb.warnOuterOverlap
          && Number(prevPbb.apronSiteX) === Number(nextPbb.apronSiteX) && Number(prevPbb.apronSiteY) === Number(nextPbb.apronSiteY);
        if (!pbbSame) {
          state.previewPbb = nextPbb;
          scheduleDraw(); drewThisMove = true;
        }
      } else {
        if (state.previewPbb) { state.previewPbb = null; scheduleDraw(); drewThisMove = true; }
      }
    } else {
      let clearedPreview = false;
      if (state.previewRemote) { state.previewRemote = null; clearedPreview = true; }
      if (state.previewTempStand) { state.previewTempStand = null; clearedPreview = true; }
      if (state.previewPbb) { state.previewPbb = null; clearedPreview = true; }
      if (state.previewHoldingPoint) { state.previewHoldingPoint = null; clearedPreview = true; }
      if (clearedPreview) { scheduleDraw(); drewThisMove = true; }
    }
    if (layoutViewIsDragging()) {
      flushLayoutTooltipRaf();
      if (flightTooltip) flightTooltip.style.display = 'none';
    } else {
      scheduleLayoutTooltipRaf(ev, wx, wy);
    }
    if (hoverChanged && !drewThisMove) { scheduleDraw(); drewThisMove = true; }
  }
  container.addEventListener('mousemove', onLayoutCanvasCoordinateMove);
  document.addEventListener('mousemove', function(ev) {
    if (!layoutSnapDragActive() && !state.dragStart && !state.isPanning) return;
    if (container && ev.target && typeof container.contains === 'function' && container.contains(ev.target)) return;
    onLayoutCanvasCoordinateMove(ev);
  }, true);
  container.addEventListener('mouseleave', function() {
    flushLayoutTooltipRaf();
    if (flightTooltip) flightTooltip.style.display = 'none';
    if (state.dragStart || state.isPanning) return;
    _layoutReadoutLastCellKey = '';
    _layoutReadoutLastPixelStr = '';
    if (cursorPixelReadoutEl) cursorPixelReadoutEl.textContent = '—';
    state.dragStart = null;
    state.isPanning = false;
    state.hoverCell = null;
    state.previewPbb = null;
    state.previewRemote = null;
    state.previewTempStand = null;
    state.previewHoldingPoint = null;
    state.apronLinkPointerWorld = null;
    state.markerRulerHoverWorld = null;
    state.markerIslandHoverWorld = null;
    state.markerAreaHoverWorld = null;
    flushDrawNow();
  });
  container.addEventListener('dblclick', function(ev) {
    if (ev.button !== 0) return;
    const rect = canvas.getBoundingClientRect();
    const sx = ev.clientX - rect.left, sy = ev.clientY - rect.top;
    const [wx, wy] = screenToWorld(sx, sy);
    if (insertSelectedVertexAt(wx, wy, !!ev.shiftKey)) {
      ev.preventDefault();
    }
  });
  function hitTestPbbEnd(wx, wy) {
    const maxD2 = (CELL_SIZE * HIT_PBB_END_CF) ** 2;
    const cands = [];
    state.pbbStands.forEach(pbb => {
      const pt = getStandConnectionPx(pbb);
      cands.push({ id: pbb.id, kind: 'pbb', x: pt[0], y: pt[1] });
    });
    state.remoteStands.forEach(st => {
      const pt = getStandAircraftMarkerWorldPxForRemoteLike(st);
      cands.push({ id: st.id, kind: 'remote', x: pt[0], y: pt[1] });
    });
    const best = findNearestItem(cands, c => [c.x, c.y], wx, wy, maxD2);
    return best || null;
  }

  function hitTestAnyTaxiwayVertex(wx, wy) {
    const click = [wx, wy];
    const maxD2 = (CELL_SIZE * TRY_PBB_MAX_EDGE_CF) ** 2;
    let best = null;
    let bestD2 = maxD2;
    state.taxiways.forEach(tw => {
      if (!tw.vertices || tw.vertices.length < 2) return;
      for (let i = 0; i < tw.vertices.length - 1; i++) {
        const [x1, y1] = cellToPixel(tw.vertices[i].col, tw.vertices[i].row);
        const [x2, y2] = cellToPixel(tw.vertices[i+1].col, tw.vertices[i+1].row);
        const near = closestPointOnSegment([x1, y1], [x2, y2], click);
        if (!near) continue;
        const d2 = dist2(near, click);
        if (d2 < bestD2) {
          bestD2 = d2;
          best = { taxiwayId: tw.id, x: near[0], y: near[1] };
        }
      }
    });
    return best;
  }

  function onLayoutContainerMouseUp(ev) {
    if (ev.button !== 0) return;
    const wasPanning = !!state.isPanning;
    flushDrawNow();
    state.isPanning = false;
    if (wasPanning) {
      const nowPerf = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
      _layoutDetailSuppressUntil = Math.max(_layoutDetailSuppressUntil, nowPerf + 120);
    }
    if (state.dragVertex) {
      state.dragVertex = null;
      return;
    }
    if (state.pathArcDrag) {
      const d = state.pathArcDrag;
      state.pathArcDrag = null;
      pushUndo();
      if (d.islandMarkerId != null) {
        const mkArc = (state.layoutMarkers || []).find(function(m) { return m && String(m.id) === String(d.islandMarkerId); });
        const nA = (mkArc && mkArc.points && mkArc.points.length) || 0;
        if (mkArc && isLayoutPolygonMarkerKind(mkArc.kind) && d.previewPx && d.vertexIndex >= 0 && d.vertexIndex < nA) {
          pathArcCommitIslandVertexFromPreview(mkArc, d.vertexIndex, d.previewPx, !!d.lastShift);
        }
      } else if (d.apronLinkId != null) {
        const lkArc = (state.apronLinks || []).find(function(l) { return l && l.id === d.apronLinkId; });
        if (lkArc && d.previewPx && typeof d.polyVertexIndex === 'number' && d.polyVertexIndex > 0) {
          pathArcCommitApronLinkFromPreview(lkArc, d.polyVertexIndex, d.previewPx, !!d.lastShift);
        }
      } else {
        const twArc = state.taxiways.find(function(t) { return t.id === d.taxiwayId; });
        if (twArc && twArc.pathType !== 'runway' && d.previewPx && d.vertexIndex > 0 && d.vertexIndex < (twArc.vertices || []).length - 1) {
          pathArcCommitFromPreview(twArc, d.vertexIndex, d.previewPx, !!d.lastShift);
        }
      }
      updatePathArcHud();
      if (typeof syncPanelFromState === 'function') syncPanelFromState();
      if (typeof updateObjectInfo === 'function') updateObjectInfo();
      if (typeof redrawLayoutAfterEdit === 'function') redrawLayoutAfterEdit();
      else {
        if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths();
        update3DSceneWhenVisible();
        draw();
      }
      return;
    }
    if (state.dragTaxiwayVertex) {
      const tw = state.taxiways.find(t => t.id === state.dragTaxiwayVertex.taxiwayId);
      if (tw && typeof syncStartEndFromVertices === 'function') syncStartEndFromVertices(tw);
      state.dragTaxiwayVertex = null;
      if (typeof syncPanelFromState === 'function') syncPanelFromState();
      if (typeof updateObjectInfo === 'function') updateObjectInfo();
      if (typeof redrawLayoutAfterEdit === 'function') redrawLayoutAfterEdit();
      else {
        if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths();
        update3DSceneWhenVisible();
        draw();
      }
      return;
    }
    if (state.dragStandRotation) {
      state.dragStandRotation = null;
      if (typeof syncPanelFromState === 'function') syncPanelFromState();
      if (typeof updateObjectInfo === 'function') updateObjectInfo();
      if (typeof redrawLayoutAfterEdit === 'function') redrawLayoutAfterEdit();
      else {
        if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths();
        update3DSceneWhenVisible();
        draw();
      }
      return;
    }
    if (state.dragPbbBridgeVertex) {
      state.dragPbbBridgeVertex = null;
      updateObjectInfo();
      draw();
      return;
    }
    if (state.dragStandConnection) {
      state.dragStandConnection = null;
      updateObjectInfo();
      if (typeof redrawLayoutAfterEdit === 'function') redrawLayoutAfterEdit();
      else {
        if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths(); else draw();
        update3DSceneWhenVisible();
      }
      return;
    }
    if (state.dragRemoteStandPosition) {
      state.dragRemoteStandPosition = null;
      if (typeof updateObjectInfo === 'function') updateObjectInfo();
      if (typeof redrawLayoutAfterEdit === 'function') redrawLayoutAfterEdit();
      else {
        if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths(); else draw();
        update3DSceneWhenVisible();
      }
      return;
    }
    if (state.dragApronLinkVertex) {
      state.dragApronLinkVertex = null;
      if (typeof updateObjectInfo === 'function') updateObjectInfo();
      if (typeof redrawLayoutAfterEdit === 'function') redrawLayoutAfterEdit();
      else {
        if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths();
        update3DSceneWhenVisible();
        draw();
      }
      return;
    }
    if (state.dragLayoutMarkerHandle) {
      const hUp = state.dragLayoutMarkerHandle;
      state.dragLayoutMarkerHandle = null;
      if (hUp.handle === 'islandVertex' && typeof hUp.vertexIndex === 'number') {
        state.selectedVertex = { type: 'layoutMarkerHandle', id: hUp.markerId, handle: 'islandVertex', vertexIndex: hUp.vertexIndex };
      } else {
        state.selectedVertex = null;
      }
      if (typeof updateObjectInfo === 'function') updateObjectInfo();
      draw();
      return;
    }
    const rect = canvas.getBoundingClientRect();
    const sx = ev.clientX - rect.left, sy = ev.clientY - rect.top;
    const [wx, wy] = screenToWorld(sx, sy);
    const placePx = worldPointToPixel(wx, wy, !!ev.shiftKey);
    const mode = settingModeSelect.value;
    const inStandDrawingMode = (mode === 'pbb' && state.pbbDrawing) || (mode === 'remote' && state.remoteDrawing) || (mode === 'tempStand' && state.tempStandDrawing) || (mode === 'holdingPoint' && state.holdingPointDrawing) || (mode === 'marker' && state.markerDrawing);
    if (!state.dragStart && !inStandDrawingMode) { state.dragStart = null; return; }
    if (handlePbbOrRemoteMouseUp2D(mode, placePx[0], placePx[1])) {
      state.dragStart = null;
      return;
    }
    if (!state.dragStart) return;
    if (!wasPanning) {
      const mode = settingModeSelect.value;
      if (mode === 'marker' && state.markerDrawing) {
        handleMarkerPlacement(wx, wy, !!ev.shiftKey);
        state.dragStart = null;
        draw();
        if (typeof syncAllocGanttSelectionHighlight === 'function') syncAllocGanttSelectionHighlight();
        return;
      }
      if (mode === 'edge') {
        rebuildDerivedGraphEdges();
        const eh = hitTestLayoutGraphEdge(wx, wy);
        if (eh) {
          state.selectedObject = { type: 'layoutEdge', id: eh.id, obj: eh };
        } else {
          state.selectedObject = null;
        }
        state.flightPathRevealFlightId = null;
        syncPanelFromState();
        updateObjectInfo();
        draw();
        if (typeof syncAllocGanttSelectionHighlight === 'function') syncAllocGanttSelectionHighlight();
        state.dragStart = null;
        return;
      }
      const hit = hitTest(wx, wy);
      let pickHit = hit;
      if (mode !== 'edge' && !(mode === 'apronTaxiway' && state.apronLinkDrawing) &&
          !state.terminalDrawingId && !state.taxiwayDrawingId) {
        const sf = hitTestSimFlightAtWorld(wx, wy);
        if (sf) pickHit = { type: 'flight', id: sf.id, obj: sf };
      }
      if (mode === 'apronTaxiway' && state.apronLinkDrawing) {
        const pbbHit = hitTestPbbEnd(wx, wy);
        const twHit = hitTestAnyTaxiwayVertex(wx, wy);
        const endpoint = pbbHit ? { kind: pbbHit.kind, standId: pbbHit.id, x: pbbHit.x, y: pbbHit.y } :
                          (twHit ? { kind: 'taxiway', taxiwayId: twHit.taxiwayId, x: twHit.x, y: twHit.y } : null);
        if (endpoint) {
          if (!state.apronLinkTemp) {
            state.apronLinkTemp = endpoint;
            state.apronLinkMidpoints = [];
          } else {
            const first = state.apronLinkTemp;
            if (first.kind !== endpoint.kind) {
              let standId, taxiwayId, tx, ty, midVertices;
              if (first.kind === 'taxiway') {
                taxiwayId = first.taxiwayId;
                standId = endpoint.standId;
                tx = first.x;
                ty = first.y;
                midVertices = (state.apronLinkMidpoints || []).slice().reverse();
              } else {
                taxiwayId = endpoint.taxiwayId;
                standId = first.standId;
                tx = endpoint.x;
                ty = endpoint.y;
                midVertices = (state.apronLinkMidpoints || []).slice();
              }
              if (standId && taxiwayId) {
                const newId = id();
                const inputName = document.getElementById('apronLinkName');
                const linkName = (inputName && String(inputName.value).trim()) || getApronLinkDefaultName(newId);
                if (findDuplicateLayoutName('apronLink', newId, linkName)) {
                  alertDuplicateLayoutName();
                } else {
                  pushUndo();
                  const linkRec = {
                    id: newId,
                    name: linkName,
                    pbbId: standId,
                    taxiwayId,
                    tx,
                    ty,
                    apronDrawFirstEndpoint: first.kind === 'taxiway' ? 'taxiway' : 'stand'
                  };
                  if (midVertices && midVertices.length) linkRec.midVertices = midVertices;
                  state.apronLinks.push(linkRec);
                  markApronLinkJunctionOverlayDirty(newId);
                  syncPanelFromState();
                  if (typeof redrawLayoutAfterEdit === 'function') redrawLayoutAfterEdit();
                  else {
                    if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths();
                    update3DSceneWhenVisible();
                  }
                }
              }
            }
            state.apronLinkTemp = null;
            state.apronLinkMidpoints = [];
            state.apronLinkPointerWorld = null;
          }
          draw();
        } else if (state.apronLinkTemp) {
          const ptCell = worldPointToCellPoint(wx, wy, !!ev.shiftKey);
          const col = ptCell.col, row = ptCell.row;
          if (col >= 0 && row >= 0 && col <= GRID_COLS && row <= GRID_ROWS) {
            const last = state.apronLinkMidpoints[state.apronLinkMidpoints.length - 1];
            if (!last || last.col !== col || last.row !== row) {
              state.apronLinkMidpoints.push({ col, row });
            }
          }
          draw();
        }
      } else if (pickHit && !layoutDrawModePreventsBackgroundObjectPick()) {
        state.flightPathRevealFlightId = null;
        state.selectedObject = pickHit;
        if (pickHit.type === 'terminal') state.currentTerminalId = pickHit.id;
        const sm = settingModeValueForHit(pickHit);
        if (sm) settingModeSelect.value = sm;
        if (pickHit.type === 'flight' && typeof switchToTab === 'function') switchToTab('flight');
        if (pickHit.type === 'layoutMarker' && typeof switchToTab === 'function') switchToTab('settings');
        if (pickHit.type === 'layoutMarker') syncMarkerSubKindTabFromSelectedLayoutMarker();
        if (typeof syncSettingsPaneToMode === 'function') syncSettingsPaneToMode();
        syncPanelFromState();
        renderObjectList();
        updateObjectInfo();
        draw();
        if (typeof syncAllocGanttSelectionHighlight === 'function') syncAllocGanttSelectionHighlight();
      } else {
        const pt = worldPointToCellPoint(wx, wy, !!ev.shiftKey);
        const col = pt.col, row = pt.row;
        if (col < 0 || row < 0 || col > GRID_COLS || row > GRID_ROWS) { state.dragStart = null; return; }
        if (mode === 'terminal') {
          if (state.terminalDrawingId) {
            let term = state.terminals.find(t => t.id === state.terminalDrawingId);
            if (!term) {
              state.terminalDrawingId = null;
              state.layoutPathDrawPointer = null;
            } else {
              const pt = { col, row };
              if (term.vertices.length === 0) {
                pushUndo();
                term.vertices.push(pt);
              } else {
                const [fx,fy] = cellToPixel(term.vertices[0].col, term.vertices[0].row);
                const d2 = dist2([fx,fy], cellToPixel(col, row));
                if (d2 < (CELL_SIZE * TERM_CLOSE_POLY_CF) ** 2 && term.vertices.length >= 3) {
                  term.closed = true;
                  state.terminalDrawingId = null;
                  state.layoutPathDrawPointer = null;
                  syncPanelFromState();
                } else {
                  const last = term.vertices[term.vertices.length-1];
                  if (last.col !== col || last.row !== row) { pushUndo(); term.vertices.push(pt); }
                }
              }
              draw();
            }
          }
        } else if (isPathLayoutMode(mode)) {
          if (state.taxiwayDrawingId) {
            const tw = state.taxiways.find(t => t.id === state.taxiwayDrawingId);
            if (tw) {
              const pt = { col, row };
              const last = tw.vertices[tw.vertices.length - 1];
              if (!last || last.col !== col || last.row !== row) {
                if (tw.pathType === 'runway' && tw.vertices.length >= 2) return;
                pushUndo();
                tw.vertices.push(pt);
                if (typeof syncStartEndFromVertices === 'function') syncStartEndFromVertices(tw);
                if (tw.pathType === 'runway' && tw.vertices.length >= 2) {
                  state.taxiwayDrawingId = null;
                  state.layoutPathDrawPointer = null;
                  syncPanelFromState();
                }
                if (typeof redrawLayoutAfterEdit === 'function') redrawLayoutAfterEdit();
                else if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths(); else draw();
              }
            }
          }
        } else if (mode === 'pbb') {
          if (tryPlacePbbAt(wx, wy)) {
            syncPanelFromState();
            draw();
          }
        } else if (mode === 'remote' && state.remoteDrawing) {
          const prev = state.previewRemote;
          if (prev && !prev.overlap && tryPlaceRemoteAt(prev.x, prev.y)) {
            syncPanelFromState();
            draw();
          }
        } else if (mode === 'tempStand' && state.tempStandDrawing) {
          const prev = state.previewTempStand;
          if (prev && !prev.overlap && tryPlaceTempStandAt(prev.x, prev.y)) {
            syncPanelFromState();
            draw();
          }
        }
      }
    }
    state.dragStart = null;
  }
  document.addEventListener('mouseup', onLayoutContainerMouseUp, true);
  let scene3d = null, camera3d = null, renderer3d = null, controls3d = null, grid3DMapper = null, raycaster3d = null, mouse3d = null, groundPlane3d = null, gridGroup3d = null;
  let mouse3dDown = null;
  const view3dContainer = document.getElementById('view3d-container');
  document.getElementById('btnView2D').classList.add('active');
  document.getElementById('btnView2D').addEventListener('click', function() {
    document.getElementById('btnView2D').classList.add('active');
    document.getElementById('btnView3D').classList.remove('active');
    document.getElementById('canvas-container').style.display = 'block';
    view3dContainer.classList.remove('active');
    if (renderer3d) renderer3d.domElement.style.display = 'none';
    requestAnimationFrame(function() {
      if (typeof resizeCanvas === 'function') resizeCanvas();
    });
  });
  document.getElementById('btnView3D').addEventListener('click', function() {
    try {
      openGrid3DViewerWindow();
    } catch (e) {
      console.error('openGrid3DViewerWindow:', e);
      alert('3D viewer failed: ' + (e && e.message ? e.message : e));
    }
  });

  function reset3DView() {
    if (!camera3d) return;
    const halfW = (GRID_COLS * CELL_SIZE) / 2;
    const halfH = (GRID_ROWS * CELL_SIZE) / 2;
    const maxDim = Math.max(halfW, halfH);
    camera3d.position.set(maxDim * 1.2, maxDim * 1.2, maxDim * 1.2);
    camera3d.lookAt(0, 0, 0);
    if (controls3d) {
      controls3d.target.set(0, 0, 0);
      controls3d.update();
    }
  }

  if (resetViewBtn) {
    resetViewBtn.addEventListener('click', function() {
      try {
        resizeCanvas();
        if (view3dContainer.classList.contains('active')) reset3DView();
        else reset2DView();
        try { draw(); } catch(e) {}
        update3DSceneWhenVisible();
      } catch (e) { console.error('Fit button error:', e); }
    });
  }
  syncLayerPopoverFromState();
  syncMapTypePopoverFromState();
  if (layerPopoverPanel) {
    layerPopoverPanel.querySelectorAll('input[data-layer-key]').forEach(function(inp) {
      inp.addEventListener('change', function() {
        const k = inp.getAttribute('data-layer-key');
        if (!k || typeof state.layers[k] !== 'boolean') return;
        state.layers[k] = !!inp.checked;
        syncLegacyViewFlagsFromLayers();
        if (k === 'image') invalidateGridUnderlay();
        syncLayerPopoverFromState();
        draw();
      });
    });
    layerPopoverPanel.querySelectorAll('input[data-layer-section-parent]').forEach(function(parentInp) {
      parentInp.addEventListener('change', function(ev) {
        ev.stopPropagation();
        const sec = parentInp.getAttribute('data-layer-section-parent');
        const keys = sec && LAYER_SECTION_KEYS[sec];
        if (!keys || !keys.length) return;
        const on = !!parentInp.checked;
        for (let i = 0; i < keys.length; i++) {
          state.layers[keys[i]] = on;
        }
        parentInp.indeterminate = false;
        syncLegacyViewFlagsFromLayers();
        invalidateGridUnderlay();
        syncLayerPopoverFromState();
        draw();
      });
    });
    layerPopoverPanel.querySelectorAll('input[data-layer-mono]').forEach(function(monoInp) {
      monoInp.addEventListener('change', function(ev) {
        ev.stopPropagation();
        const mk = monoInp.getAttribute('data-layer-mono');
        if (!mk || !state.layerMono || typeof state.layerMono[mk] !== 'boolean') return;
        state.layerMono[mk] = !!monoInp.checked;
        syncLayerPopoverFromState();
        draw();
      });
    });
  }
  const btnLayerPopoverAll = document.getElementById('btnLayerPopoverAll');
  if (btnLayerPopoverAll) {
    btnLayerPopoverAll.addEventListener('click', function(ev) {
      ev.stopPropagation();
      const allOn = LAYER_STATE_KEYS.every(function(k) { return !!state.layers[k]; });
      const next = !allOn;
      for (let i = 0; i < LAYER_STATE_KEYS.length; i++) {
        state.layers[LAYER_STATE_KEYS[i]] = next;
      }
      syncLegacyViewFlagsFromLayers();
      invalidateGridUnderlay();
      syncLayerPopoverFromState();
      draw();
    });
  }
  function toggleShowLayoutMarkersFromUi() {
    const next = !(state.layers.textRuler && state.layers.dummyFlight);
    state.layers.textRuler = next;
    state.layers.dummyFlight = next;
    syncLegacyViewFlagsFromLayers();
    syncLayerPopoverFromState();
    draw();
  }
  const btnLayoutMarkersToggle = document.getElementById('btnLayoutMarkersToggle');
  const btnGridMarkerOverlayToggle = document.getElementById('btnGridMarkerOverlayToggle');
  if (btnLayoutMarkersToggle) btnLayoutMarkersToggle.addEventListener('click', toggleShowLayoutMarkersFromUi);
  if (btnGridMarkerOverlayToggle) btnGridMarkerOverlayToggle.addEventListener('click', toggleShowLayoutMarkersFromUi);
  if (layerPopoverBtn && layerPopoverPanel) {
    layerPopoverBtn.addEventListener('click', function(ev) {
      ev.stopPropagation();
      const open = layerPopoverPanel.hasAttribute('hidden');
      setLayerPopoverOpen(open);
    });
    document.addEventListener('click', function(ev) {
      if (!layerPopoverWrap || layerPopoverPanel.hasAttribute('hidden')) return;
      if (layerPopoverWrap.contains(ev.target)) return;
      setLayerPopoverOpen(false);
    });
  }
  const btnHeatmapToggle = document.getElementById('btnHeatmapToggle');
  if (btnHeatmapToggle) {
    btnHeatmapToggle.addEventListener('click', function(ev) {
      ev.stopPropagation();
      if (!state.hasSimulationResult) return;
      state.mapTypeMode = (state.mapTypeMode === 'heatmap') ? 'normal' : 'heatmap';
      syncMapTypePopoverFromState();
      safeDraw({ bypassSimScrubGuard: true });
      if (typeof update3DSceneWhenVisible === 'function') update3DSceneWhenVisible();
    });
  }
  const colorPopoverBtn = document.getElementById('btnColorPopover');
  const colorPopoverPanel = document.getElementById('colorPopoverPanel');
  const colorPopoverWrap = document.getElementById('colorPopoverWrap');
  const flightSimColorModeSel = document.getElementById('flightSimColorMode');
  function setColorPopoverOpen(open) {
    if (!colorPopoverPanel || !colorPopoverBtn) return;
    if (open) {
      colorPopoverPanel.removeAttribute('hidden');
      colorPopoverBtn.setAttribute('aria-expanded', 'true');
    } else {
      colorPopoverPanel.setAttribute('hidden', '');
      colorPopoverBtn.setAttribute('aria-expanded', 'false');
    }
  }
  if (flightSimColorModeSel) {
    flightSimColorModeSel.value = state.flightColorMode || 'all';
    flightSimColorModeSel.addEventListener('change', function() {
      state.flightColorMode = String(this.value || 'all');
      if (typeof draw === 'function') draw();
    });
  }
  if (colorPopoverBtn && colorPopoverPanel) {
    colorPopoverBtn.addEventListener('click', function(ev) {
      ev.stopPropagation();
      const open = colorPopoverPanel.hasAttribute('hidden');
      if (open && layerPopoverPanel && !layerPopoverPanel.hasAttribute('hidden')) setLayerPopoverOpen(false);
      setColorPopoverOpen(open);
    });
    document.addEventListener('click', function(ev) {
      if (!colorPopoverWrap || colorPopoverPanel.hasAttribute('hidden')) return;
      if (colorPopoverWrap.contains(ev.target)) return;
      setColorPopoverOpen(false);
    });
  }
  const aiAssistantDock = document.getElementById('aiAssistantDock');
  const btnAiAssistantDockClose = document.getElementById('btnAiAssistantDockClose');
  const aiModeToggleEls = document.querySelectorAll('[data-ai-mode-toggle]');
  function setAiAssistantDockOpen(open) {
    state.aiAssistantDockOpen = !!open;
    if (aiAssistantDock) {
      if (state.aiAssistantDockOpen) {
        aiAssistantDock.removeAttribute('hidden');
        aiAssistantDock.classList.add('is-open');
        aiAssistantDock.setAttribute('aria-hidden', 'false');
      } else {
        aiAssistantDock.setAttribute('hidden', '');
        aiAssistantDock.classList.remove('is-open');
        aiAssistantDock.setAttribute('aria-hidden', 'true');
      }
    }
    aiModeToggleEls.forEach(function(el) {
      el.classList.toggle('active', state.aiAssistantDockOpen);
      el.setAttribute('aria-pressed', state.aiAssistantDockOpen ? 'true' : 'false');
    });
  }
  if (aiAssistantDock && aiModeToggleEls.length) {
    aiModeToggleEls.forEach(function(el) {
      el.addEventListener('click', function() {
        setAiAssistantDockOpen(!state.aiAssistantDockOpen);
      });
    });
  }
  if (btnAiAssistantDockClose) {
    btnAiAssistantDockClose.addEventListener('click', function() {
      setAiAssistantDockOpen(false);
    });
  }
  const aiAssistantDockThread = document.getElementById('aiAssistantDockThread');
  const aiAssistantComposerInput = document.getElementById('aiAssistantComposerInput');
  const btnAiAssistantSend = document.getElementById('btnAiAssistantSend');
  const _aiKimiChatHistory = [];
  function appendAiAssistantThread(role, text) {
    if (!aiAssistantDockThread) return;
    const wrap = document.createElement('div');
    wrap.className = 'ai-assistant-msg ' + (role === 'user' ? 'ai-assistant-msg-user' : 'ai-assistant-msg-agent');
    const meta = document.createElement('div');
    meta.className = 'ai-assistant-msg-meta';
    meta.textContent = role === 'user' ? 'You' : 'Kimi';
    const body = document.createElement('div');
    body.className = 'ai-assistant-msg-body';
    body.textContent = text;
    wrap.appendChild(meta);
    wrap.appendChild(body);
    aiAssistantDockThread.appendChild(wrap);
    aiAssistantDockThread.scrollTop = aiAssistantDockThread.scrollHeight;
  }
  function setAiAssistantSendBusy(busy) {
    if (btnAiAssistantSend) {
      btnAiAssistantSend.disabled = !!busy;
      btnAiAssistantSend.setAttribute('aria-busy', busy ? 'true' : 'false');
    }
    if (aiAssistantComposerInput) aiAssistantComposerInput.disabled = !!busy;
  }
  function sendAiAssistantKimiMessage() {
    if (!aiAssistantComposerInput) return;
    const text = (aiAssistantComposerInput.value || '').trim();
    if (!text) return;
    const apiBase = (typeof getLayoutApiBase === 'function') ? getLayoutApiBase() : (LAYOUT_API_URL || '');
    if (!apiBase) {
      appendAiAssistantThread('agent', 'Layout API base URL is not set — cannot call /api/ai-chat.');
      return;
    }
    aiAssistantComposerInput.value = '';
    appendAiAssistantThread('user', text);
    _aiKimiChatHistory.push({ role: 'user', content: text });
    setAiAssistantSendBusy(true);
    fetch(apiBase + '/api/ai-chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: _aiKimiChatHistory.slice(), model: 'kimi-k2.5' }),
    }).then(function(r) {
      return r.text().then(function(t) {
        let j = null;
        try {
          j = t ? JSON.parse(t) : null;
        } catch (e) {
          j = { ok: false, error: (t || '').slice(0, 240) };
        }
        return { ok: r.ok, status: r.status, j: j };
      });
    }).then(function(o) {
      setAiAssistantSendBusy(false);
      if (!o.ok || !o.j || o.j.ok === false) {
        const err = (o.j && (o.j.hint || o.j.error)) || ('HTTP ' + o.status);
        appendAiAssistantThread('agent', String(err));
        _aiKimiChatHistory.pop();
        return;
      }
      const reply = (o.j && o.j.reply) != null ? String(o.j.reply) : '';
      appendAiAssistantThread('agent', reply || '(empty)');
      _aiKimiChatHistory.push({ role: 'assistant', content: reply });
    }).catch(function(e) {
      setAiAssistantSendBusy(false);
      appendAiAssistantThread('agent', (e && e.message) ? e.message : 'Network error');
      _aiKimiChatHistory.pop();
    });
  }
  if (btnAiAssistantSend && aiAssistantComposerInput) {
    btnAiAssistantSend.addEventListener('click', sendAiAssistantKimiMessage);
    aiAssistantComposerInput.addEventListener('keydown', function(ev) {
      if (ev.key === 'Enter' && !ev.shiftKey) {
        ev.preventDefault();
        sendAiAssistantKimiMessage();
      }
    });
  }
  const btnDesignerPageUpdate = document.getElementById('btnDesignerPageUpdate');
  if (btnDesignerPageUpdate) {
    btnDesignerPageUpdate.addEventListener('click', function() {
      if (typeof syncStateFromPanel === 'function') syncStateFromPanel();
      if (typeof applyPathGraphSyncNow === 'function') applyPathGraphSyncNow();
      if (typeof renderObjectList === 'function') renderObjectList();
      if (typeof updateObjectInfo === 'function') updateObjectInfo();
      if (typeof triggerArrivalConfigResampleFromLayoutEdit === 'function') triggerArrivalConfigResampleFromLayoutEdit();
      if (typeof draw === 'function') draw();
      if (typeof update3DSceneWhenVisible === 'function') update3DSceneWhenVisible();
      if (typeof markDesignerPageUpdateFresh === 'function') markDesignerPageUpdateFresh();
      /** Path graph is fresh but last Pro Sim no longer matches (e.g. arrival RET / layout resample). */
      if (typeof markProSimSyncStaleFromSchedule === 'function') markProSimSyncStaleFromSchedule();
      if (typeof draw === 'function') draw();
    });
  }
  if (typeof syncProSimButtonFromDesignerPageState === 'function') syncProSimButtonFromDesignerPageState();
  class Grid3DMapper {
    constructor(cols, rows, cellSize) {
      this.cols = cols;
      this.rows = rows;
      this.cellSize = cellSize;
      this.ox = (cols * cellSize) / 2;
      this.oz = (rows * cellSize) / 2;
    }
    pixelToWorldXZ(x, y) {
      return { x: this.ox - x, z: this.oz - y };
    }
    cellToWorld(col, row, height) {
      const [px, py] = cellToPixel(col, row);
      const p = this.pixelToWorldXZ(px, py);
      return new THREE.Vector3(p.x, height, p.z);
    }
    worldFromPixel(x, y, height) {
      const p = this.pixelToWorldXZ(x, y);
      return new THREE.Vector3(p.x, height, p.z);
    }
    shapeFromCell(col, row) {
      const [px, py] = cellToPixel(col, row);
      return { x: this.ox - px, y: py - this.oz };
    }
    worldToPixel(xWorld, zWorld) {
      return { x: this.ox - xWorld, y: this.oz - zWorld };
    }
    worldToCell(xWorld, zWorld) {
      const p = this.worldToPixel(xWorld, zWorld);
      let col = Math.round(p.x / this.cellSize);
      let row = Math.round(p.y / this.cellSize);
      col = Math.max(0, Math.min(this.cols, col));
      row = Math.max(0, Math.min(this.rows, row));
      return [col, row];
    }
  }

  function init3D() {
    if (renderer3d) { renderer3d.domElement.style.display = 'block'; update3DScene(); return; }
    const w = view3dContainer.clientWidth, h = view3dContainer.clientHeight;
    scene3d = new THREE.Scene();
    scene3d.background = new THREE.Color(GRID_VIEW_BG);
    gridGroup3d = new THREE.Group();
    scene3d.add(gridGroup3d);
    camera3d = new THREE.PerspectiveCamera(50, w/h, 1, 100000);
    const halfW = (GRID_COLS * CELL_SIZE) / 2, halfH = (GRID_ROWS * CELL_SIZE) / 2;
    const maxDim = Math.max(halfW, halfH);
    camera3d.position.set(maxDim * 1.2, maxDim * 1.2, maxDim * 1.2);
    camera3d.lookAt(0, 0, 0);
    const axisLen = CELL_SIZE * 8;
    const axisOrigin = new THREE.Vector3(-maxDim, 0, -maxDim);
    function addAxis(toVec, color) {
      const pts = [axisOrigin, axisOrigin.clone().add(toVec)];
      const geo = new THREE.BufferGeometry().setFromPoints(pts);
      const mat = new THREE.LineBasicMaterial({ color });
      const line = new THREE.Line(geo, mat);
      gridGroup3d.add(line);
    }
    addAxis(new THREE.Vector3(axisLen, 0, 0), 0xef4444);
    addAxis(new THREE.Vector3(0, 0, axisLen), 0x22c55e);
    addAxis(new THREE.Vector3(0, axisLen, 0), 0x7c6af7);
    function createAxisLabel(text, color, endVec) {
      const size = 128;
      const canvasLabel = document.createElement('canvas');
      canvasLabel.width = size;
      canvasLabel.height = size;
      const g = canvasLabel.getContext('2d');
      g.clearRect(0, 0, size, size);
      g.font = 'bold 72px system-ui';
      g.fillStyle = color;
      g.textAlign = 'center';
      g.textBaseline = 'middle';
      g.fillText(text, size / 2, size / 2);
      const tex = new THREE.CanvasTexture(canvasLabel);
      const mat = new THREE.SpriteMaterial({ map: tex, transparent: true });
      const sprite = new THREE.Sprite(mat);
      const s = CELL_SIZE * 3;
      sprite.scale.set(s, s, 1);
      sprite.position.copy(axisOrigin.clone().add(endVec));
      gridGroup3d.add(sprite);
    }
    createAxisLabel('x', '#ef4444', new THREE.Vector3(axisLen * 1.1, 0, 0));
    createAxisLabel('y', '#22c55e', new THREE.Vector3(0, 0, axisLen * 1.1));
    createAxisLabel('z', '#7c6af7', new THREE.Vector3(0, axisLen * 1.1, 0));
    grid3DMapper = new Grid3DMapper(GRID_COLS, GRID_ROWS, CELL_SIZE);
    renderer3d = new THREE.WebGLRenderer({ antialias: true });
    renderer3d.setSize(w, h);
    renderer3d.setPixelRatio(window.devicePixelRatio || 1);
    view3dContainer.appendChild(renderer3d.domElement);
    controls3d = new THREE.OrbitControls(camera3d, renderer3d.domElement);
    controls3d.enableDamping = true;
    controls3d.dampingFactor = 0.05;
    raycaster3d = new THREE.Raycaster();
    mouse3d = new THREE.Vector2();
    groundPlane3d = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
    const dom3d = renderer3d.domElement;
    function getHitPoint(ev) {
      const rect = dom3d.getBoundingClientRect();
      const ndcX = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
      const ndcY = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
      mouse3d.set(ndcX, ndcY);
      raycaster3d.setFromCamera(mouse3d, camera3d);
      const hit = new THREE.Vector3();
      return raycaster3d.ray.intersectPlane(groundPlane3d, hit) ? hit : null;
    }
    dom3d.addEventListener('mousedown', function(ev) {
      if (ev.button === 0) mouse3dDown = { x: ev.clientX, y: ev.clientY };
    });
    dom3d.addEventListener('mouseup', function(ev) {
      if (ev.button !== 0 || !mouse3dDown) return;
      const dx = ev.clientX - mouse3dDown.x, dy = ev.clientY - mouse3dDown.y;
      if (dx*dx + dy*dy > 25) { mouse3dDown = null; return; }
      mouse3dDown = null;
      const hit = getHitPoint(ev);
      if (!hit || !grid3DMapper) return;
      const mode = settingModeSelect.value;
      const p = grid3DMapper.worldToPixel(hit.x, hit.z);
      const wx = p.x, wy = p.y;
      const [col, row] = grid3DMapper.worldToCell(hit.x, hit.z);
      tryCommitStandPlacement3D(mode, wx, wy, col, row);
    });
    const step = CELL_SIZE;
    const faintLines = [];
    const majorLines = [];
    let kx = 0;
    for (let x = -maxDim; x <= maxDim; x += step, kx++) {
      const pts = [new THREE.Vector3(x, 0, -maxDim), new THREE.Vector3(x, 0, maxDim)];
      if (kx % GRID_MAJOR_INTERVAL === 0) majorLines.push.apply(majorLines, pts);
      else faintLines.push.apply(faintLines, pts);
    }
    let kz = 0;
    for (let z = -maxDim; z <= maxDim; z += step, kz++) {
      const pts = [new THREE.Vector3(-maxDim, 0, z), new THREE.Vector3(maxDim, 0, z)];
      if (kz % GRID_MAJOR_INTERVAL === 0) majorLines.push.apply(majorLines, pts);
      else faintLines.push.apply(faintLines, pts);
    }
    if (faintLines.length) {
      const faintGeo = new THREE.BufferGeometry().setFromPoints(faintLines);
      const faintMat = new THREE.LineBasicMaterial({
        color: 0xd4d4d4,
        transparent: true,
        opacity: 0.2,
        depthTest: false
      });
      gridGroup3d.add(new THREE.LineSegments(faintGeo, faintMat));
    }
    if (majorLines.length) {
      const majorGeo = new THREE.BufferGeometry().setFromPoints(majorLines);
      const majorMat = new THREE.LineBasicMaterial({
        color: 0xffffff,
        transparent: true,
        opacity: 0.35,
        depthTest: false
      });
      gridGroup3d.add(new THREE.LineSegments(majorGeo, majorMat));
    }
    update3DScene();
  }

  function update3DScene() {
    if (!scene3d) return;
    while (scene3d.children.length > 1) scene3d.remove(scene3d.children[scene3d.children.length - 1]);
    if (!grid3DMapper) grid3DMapper = new Grid3DMapper(GRID_COLS, GRID_ROWS, CELL_SIZE);
  }

  function update3DSceneWhenVisible() {
    if (typeof update3DScene !== 'function') return;
    if (!view3dContainer || !view3dContainer.classList.contains('active')) return;
    update3DScene();
  }

  function animate3D() {
    if (!renderer3d || !view3dContainer.classList.contains('active')) return;
    requestAnimationFrame(animate3D);
    if (controls3d) controls3d.update();
    if (renderer3d && scene3d && camera3d) renderer3d.render(scene3d, camera3d);
  }

  container.addEventListener('wheel', function(ev) {
    ev.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const mx = ev.clientX - rect.left, my = ev.clientY - rect.top;
    const wx = (mx - state.panX) / Math.max(state.scale, 1e-9), wy = (my - state.panY) / Math.max(state.scale, 1e-9);
    let dy = ev.deltaY;
    if (ev.deltaMode === 1) dy *= 16;
    else if (ev.deltaMode === 2) dy *= 120;
    const step = dy < 0 ? 1.15 : (1 / 1.15);
    state.scale *= step;
    state.scale = Math.max(CANVAS_MIN_ZOOM, Math.min(CANVAS_MAX_ZOOM, state.scale));
    state.panX = mx - wx * state.scale;
    state.panY = my - wy * state.scale;
    const nowPerf = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
    _layoutDetailSuppressUntil = nowPerf + 200;
    scheduleDraw();
  }, { passive: false });

  window.addEventListener('resize', function() {
    resizeCanvas();
    if (renderer3d && view3dContainer.classList.contains('active')) {
      const w = view3dContainer.clientWidth, h = view3dContainer.clientHeight;
      camera3d.aspect = w / h;
      camera3d.updateProjectionMatrix();
      renderer3d.setSize(w, h);
    }
  });
  try { applyInitialLayoutFromJson(); } catch(applyErr) { console.error('Layout apply failed:', applyErr); }
  updateLayoutNameBar(INITIAL_LAYOUT_DISPLAY_NAME || 'default_layout');
  resizeCanvas();
  reset2DView();
  syncPanelFromState();
  if (typeof draw === 'function') draw();
  update3DSceneWhenVisible();
  if (typeof renderKpiDashboard === 'function') renderKpiDashboard('Initial load');
})();
