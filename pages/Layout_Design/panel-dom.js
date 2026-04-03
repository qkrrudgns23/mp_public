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

  const PRO_SIM_PHASE_Z = { Landing: 0, Arr_taxi: 1, Dep_taxi: 2 };
  function proSimPhaseStrokeStyle(phaseRaw) {
    const p = (phaseRaw != null && String(phaseRaw).trim()) ? String(phaseRaw).trim() : 'Landing';
    if (p === 'Arr_taxi') {
      return { wMul: 1.72, stroke: '#3b82f6' };
    }
    if (p === 'Dep_taxi') {
      return { wMul: 0.58, stroke: '#ef4444' };
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
        ctx.shadowColor = c2dObjectSelectedGlow();
        ctx.shadowBlur = c2dObjectSelectedGlowBlur();
        ctx.stroke();
        ctx.restore();
      }
      ctx.save();
      ctx.translate(x, y);
      const ang = Math.atan2(ny, nx);
      ctx.rotate(ang);
      ctx.fillStyle = apron2DGlyphFill();
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
        ctx.strokeStyle = 'rgba(15,23,42,0.85)';
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
        if (typeof update3DScene === 'function') update3DScene();
      }
      window.requestAnimationFrame(tick);
    }
    window.requestAnimationFrame(tick);
  }

  const AIRCRAFT_TYPES = (typeof INFORMATION === 'object' && INFORMATION && INFORMATION.tiers && INFORMATION.tiers.aircraft && Array.isArray(INFORMATION.tiers.aircraft.types)) ? INFORMATION.tiers.aircraft.types : [];
  const AIRCRAFT_BY_ID = {};
  AIRCRAFT_TYPES.forEach(a => { AIRCRAFT_BY_ID[a.id || a.name] = a; });
  function getAircraftInfoByType(typeId) {
    return AIRCRAFT_BY_ID[typeId] || null;
  }
  function getCodeForAircraft(typeId) {
    const a = getAircraftInfoByType(typeId);
    return (a && a.icao) ? a.icao : 'C';
  }
  function populateAircraftSelect(sel) {
    if (!sel) return;
    const opts = AIRCRAFT_TYPES.map(a => '<option value="' + escapeHtml(String(a.id || a.name || '')) + '">' + escapeHtml(a.name || a.id || '') + '</option>').join('');
    sel.innerHTML = opts || '<option value="A320">Airbus A320</option>';
    if (!opts && sel.options.length) sel.value = 'A320';
    else if (sel.options.length) sel.value = sel.options[0].value;
  }
  function getAircraftConstraintOptions() {
    return AIRCRAFT_TYPES.map(function(a) {
      const id = String(a.id || a.name || '').trim();
      const label = String(a.name || a.id || id || '').trim();
      return { id: id, label: label || id };
    }).filter(function(item) { return !!item.id; });
  }
  function normalizeStandCategoryMode(rawMode, fallbackMode) {
    const mode = String(rawMode || fallbackMode || 'icao').trim().toLowerCase();
    return mode === 'aircraft' ? 'aircraft' : 'icao';
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
    const fallback = isRemote ? (_remoteTier.defaultCategoryMode || 'icao') : (_pbbTier.defaultCategoryMode || 'icao');
    return normalizeStandCategoryMode(stand && stand.categoryMode, fallback);
  }
  function getStandAllowedAircraftTypes(stand) {
    return normalizeAllowedAircraftTypes(stand && stand.allowedAircraftTypes);
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
    return normalizeAngleDeg(getPBBStandAngle(pbb) * 180 / Math.PI);
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
    const standSize = getStandSizeMeters((pbb && pbb.category) || 'C');
    const dist = getPbbLengthMeters(pbb) + Math.max(standSize * 0.55, 10);
    return [origin[0] + Math.cos(safeAngle) * dist, origin[1] + Math.sin(safeAngle) * dist];
  }
  function getRemoteRotationHandlePx(st) {
    const center = getRemoteStandCenterPx(st);
    const angle = getRemoteStandAngleRad(st);
    const standSize = getStandSizeMeters((st && st.category) || 'C');
    const dist = (standSize * 0.5) + Math.max(standSize * 0.35, 10);
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
    const length = getPbbLengthMeters(pbb);
    const angle = getPBBStandAngle(pbb);
    const dirX = Math.cos(angle), dirY = Math.sin(angle);
    const tanX = -dirY, tanY = dirX;
    const standSize = getStandSizeMeters((pbb && pbb.category) || 'C');
    const spread = Math.min(Math.max(standSize * 0.18, 4), standSize * 0.4);
    const offsetIndex = bridgeIndex - (count - 1) / 2;
    const lateral = spread * offsetIndex;
    const startX = Number(pbb.x1 || 0) + tanX * lateral;
    const startY = Number(pbb.y1 || 0) + tanY * lateral;
    const endX = Number(pbb.x2 || 0) + tanX * (lateral * 0.55);
    const endY = Number(pbb.y2 || 0) + tanY * (lateral * 0.55);
    const midX = startX + dirX * (length * 0.45);
    const midY = startY + dirY * (length * 0.45);
    return [
      { x: startX, y: startY },
      { x: midX, y: midY },
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
    if (pbb.apronSiteX == null || pbb.apronSiteY == null) {
      pbb.apronSiteX = Number(pbb.x2 || 0);
      pbb.apronSiteY = Number(pbb.y2 || 0);
    }
  }
  function setPbbGeometryFromAngleLength(pbb, angleDeg, lengthMeters, resetBridgeGeometry) {
    const ang = normalizeAngleDeg(angleDeg);
    const len = Math.max(1, Number(lengthMeters) || 1);
    const rad = ang * Math.PI / 180;
    const anchor = getPbbAnchorPx(pbb);
    pbb.x1 = anchor[0];
    pbb.y1 = anchor[1];
    pbb.x2 = anchor[0] + Math.cos(rad) * len;
    pbb.y2 = anchor[1] + Math.sin(rad) * len;
    pbb.angleDeg = ang;
    if (resetBridgeGeometry !== false) {
      delete pbb.pbbBridges;
    }
    rebuildPbbBridgeGeometry(pbb);
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
    pbb.categoryMode = getStandCategoryMode(pbb);
    pbb.allowedAircraftTypes = getStandAllowedAircraftTypes(pbb);
    pbb.pbbCount = Math.max(1, Math.min(8, parseInt(pbb.pbbCount != null ? pbb.pbbCount : (_pbbTier.defaultBridgeCount || 1), 10) || 1));
    if (pbb.x1 != null && pbb.y1 != null && pbb.x2 != null && pbb.y2 != null) {
      pbb.angleDeg = pbb.angleDeg != null
        ? normalizeAngleDeg(pbb.angleDeg)
        : normalizeAngleDeg(Math.atan2((Number(pbb.y2) || 0) - (Number(pbb.y1) || 0), (Number(pbb.x2) || 0) - (Number(pbb.x1) || 0)) * 180 / Math.PI);
      rebuildPbbBridgeGeometry(pbb);
    }
    return pbb;
  }
  function normalizeRemoteStandObject(rawStand) {
    const stand = Object.assign({}, rawStand || {});
    stand.categoryMode = getStandCategoryMode(stand);
    stand.allowedAircraftTypes = getStandAllowedAircraftTypes(stand);
    stand.angleDeg = normalizeAngleDeg(stand.angleDeg != null ? stand.angleDeg : 0);
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
    const aircraftEl = document.getElementById('flightAircraftType');
    const regEl = document.getElementById('flightReg');
    const layoutNameInput = document.getElementById('layoutName');
    const saveLayoutBtn = document.getElementById('btnSaveLayout');
    const layoutMsgEl = document.getElementById('layoutMessage');
    const layoutLoadListEl = document.getElementById('layoutLoadList');
    const globalUpdateBtn = document.getElementById('btnGlobalUpdate');
    if (!arrDepEl) return;
    populateAircraftSelect(aircraftEl);

    function randomAirlineCode() { return DEFAULT_AIRLINE_CODES[Math.floor(Math.random() * DEFAULT_AIRLINE_CODES.length)]; }
    function randomFlightNumber(airlineCode) { return (airlineCode || randomAirlineCode()) + String(Math.floor(1000 + Math.random() * 9000)); }
    function getDefaultSibtMinutes() {
      let maxT = 0;


      (state.flights || []).forEach(f => {
        if (!f) return;
        const sibt = f.sibtMin_d != null ? f.sibtMin_d : (typeof f.timeMin === 'number' ? f.timeMin : 0);
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
          if (typeof alert === 'function') alert(m);
          const ab = document.getElementById('btnApplySimResult');
          if (ab) ab.disabled = true;
        }
        const base = proSimApiBase();
        if (!base) {
          failProSim('Layout API가 설정되지 않았습니다. run_app.py로 서버를 띄운 뒤 다시 시도하세요.');
          return;
        }
        if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
        if (typeof clearAllFlightTimelines === 'function') clearAllFlightTimelines();
        const applyBtnEl = document.getElementById('btnApplySimResult');
        const playDockBtnEl = document.getElementById('btnShowPlayDock');
        if (applyBtnEl) applyBtnEl.disabled = true;
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
        try {
          layoutPayload = serializeCurrentLayout();
        } catch (e1) {
          failProSim(e1 && e1.message);
          return;
        }
        if (typeof setGlobalUpdateProgressUi === 'function') {
          setGlobalUpdateProgressUi(true, 'airside_sim 시작…', 3);
        }
        fetch(base + '/api/run-simulation', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ layout: layoutPayload, layoutName: layoutName, name: layoutName }),
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
                  if (typeof setGlobalUpdateProgressUi === 'function') {
                    setGlobalUpdateProgressUi(true, 'Airside DES (utils/airside_sim) 실행 중…', pct);
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
