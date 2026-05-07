          const [x2, y2] = cellToPixel(tw.vertices[j + 1].col, tw.vertices[j + 1].row);
          const near = closestPointOnSegment([x1, y1], [x2, y2], click);
          if (near && dist2(near, click) < pathCenterlineHitD2) return { type: 'taxiway', id: tw.id, obj: tw };
        }
      }
    }
    if (layoutMarkersVisible()) {
      const arHit = hitTestLayoutMarker(wx, wy, { onlyKind: 'area' });
      if (arHit) return arHit;
    }
    return null;
  }

  function hitTestSimFlightAtWorld(wx, wy) {
    if (!simPlaybackVisualsActive() || simPlaybackHeavyVisualsSuppressed() || !state.flights || !state.flights.length) return null;
    const tSec = state.simTimeSec;
    let best = null;
    let bestD2 = Infinity;
    const flights = state.flights;
    for (let i = 0; i < flights.length; i++) {
      const f = flights[i];
      if (!f || flightBlockedLikeNoWay(f)) continue;
      const pose = getFlightPoseAtTimeForDraw(f, tSec);
      if (!pose) continue;
      const dx = pose.x - wx, dy = pose.y - wy;
      const d2 = dx * dx + dy * dy;
      const poly = simFlightSilhouetteWorldPolygon(f, pose, tSec);
      if (poly.length >= 3 && pointInPolygonXY([wx, wy], poly) && d2 < bestD2) {
        bestD2 = d2;
        best = f;
      }
    }
    return best;
  }

  function hitTestTerminalVertex(wx, wy) {
    const maxD2 = (CELL_SIZE * HIT_TERM_VTX_CF) ** 2;
    const cands = [];
    state.terminals.forEach(t => {
      t.vertices.forEach((v, idx) => {
        cands.push({ terminalId: t.id, index: idx, v });
      });
    });
    const best = findNearestItem(cands, c => cellToPixel(c.v.col, c.v.row), wx, wy, maxD2);
    return best ? { terminalId: best.terminalId, index: best.index } : null;
  }

  function hitTestTaxiwayVertex(wx, wy) {
    if (!state.selectedObject || state.selectedObject.type !== 'taxiway') return null;
    const tw = state.selectedObject.obj;
    if (!tw || !tw.vertices || tw.vertices.length === 0) return null;
    const click = [wx, wy];
    const maxD2 = (CELL_SIZE * HIT_TW_VTX_CF) ** 2;
    let best = null;
    let bestD2 = maxD2;
    tw.vertices.forEach((v, idx) => {
      const [vx, vy] = cellToPixel(v.col, v.row);
      const d2 = dist2([vx, vy], click);
      if (d2 < bestD2) {
        bestD2 = d2;
        best = { taxiwayId: tw.id, index: idx };
      }
    });
    return best;
  }
  function hitTestPbbEditablePoint(wx, wy) {
    if (!state.selectedObject || state.selectedObject.type !== 'pbb') return null;
    const pbb = state.selectedObject.obj;
    if (!pbb || pbb.id !== state.selectedObject.id) return null;
    const click = [wx, wy];
    const maxD2 = (CELL_SIZE * HIT_PBB_END_CF) ** 2;
    let best = null;
    let bestD2 = maxD2;
    (Array.isArray(pbb.pbbBridges) ? pbb.pbbBridges : []).forEach(function(bridge, bridgeIdx) {
      (Array.isArray(bridge.points) ? bridge.points : []).forEach(function(pt, ptIdx) {
        if (ptIdx === 1) return;
        const d2 = dist2([Number(pt.x) || 0, Number(pt.y) || 0], click);
        if (d2 < bestD2) {
          bestD2 = d2;
          best = { type: 'bridge', bridgeIndex: bridgeIdx, pointIndex: ptIdx };
        }
      });
    });
    const apronPt = getStandAircraftMarkerWorldPxForPbb(pbb);
    const apronD2 = dist2(apronPt, click);
    if (apronD2 < bestD2) best = { type: 'apronSite' };
    return best;
  }
  function hitTestRemoteStandDragPoint(wx, wy) {
    if (!state.selectedObject || state.selectedObject.type !== 'remote') return null;
    const st = state.selectedObject.obj;
    if (!st || st.id !== state.selectedObject.id) return null;
    const click = [wx, wy];
    const maxD2 = (CELL_SIZE * HIT_PBB_END_CF) ** 2;
    const mk = getStandAircraftMarkerWorldPxForRemoteLike(st);
    if (dist2(mk, click) <= maxD2) return { type: 'remoteCenter' };
    return null;
  }
  function findInsertSegment(vertices, closed, wx, wy) {
    if (!Array.isArray(vertices) || vertices.length < 2) return null;
    const click = [wx, wy];
    const maxD2 = (CELL_SIZE * INSERT_VERTEX_HIT_CF) ** 2;
    let best = null;
    let bestD2 = maxD2;
    const lastSeg = closed ? vertices.length : (vertices.length - 1);
    function vertexToPixel(v) {
      if (Array.isArray(v) && v.length >= 2) return [Number(v[0]) || 0, Number(v[1]) || 0];
      if (v && v.x != null && v.y != null) return [Number(v.x) || 0, Number(v.y) || 0];
      return cellToPixel(v.col, v.row);
    }
    for (let i = 0; i < lastSeg; i++) {
      const curr = vertices[i];
      const next = vertices[(i + 1) % vertices.length];
      const p1 = vertexToPixel(curr);
      const p2 = vertexToPixel(next);
      const near = closestPointOnSegment(p1, p2, click);
      if (!near) continue;
      const d2 = dist2(near, click);
      if (d2 < bestD2) {
        bestD2 = d2;
        best = { insertIndex: i + 1, near: near };
      }
    }
    return best;
  }
  const PATH_ARC_MIN_BULGE_PX = 2;
  const PATH_ARC_MAX_BULGE_FRAC = 0.45;
  function pathArcAngleDiffCCW(t0, t1) {
    let d = t1 - t0;
    while (d < 0) d += 2 * Math.PI;
    while (d >= 2 * Math.PI) d -= 2 * Math.PI;
    return d;
  }
  function pathArcPointBetweenAnglesCCW(tStart, tProbe, spanCCW) {
    return pathArcAngleDiffCCW(tStart, tProbe) <= spanCCW + 1e-10;
  }
  function pathArcCircumcircle(ax, ay, bx, by, cx, cy) {
    const d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
    if (Math.abs(d) < 1e-12) return null;
    const a2 = ax * ax + ay * ay;
    const b2 = bx * bx + by * by;
    const c2 = cx * cx + cy * cy;
    const ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d;
    const uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d;
    const r = Math.hypot(ax - ux, ay - uy);
    if (!(r > 1e-9)) return null;
    return { ox: ux, oy: uy, r: r };
  }
  /** Endpoints A,B and point C on arc; returns world px polyline A→B along the circle through C. */
  function pathArcSampleThreePointWorldPx(ax, ay, bx, by, cx, cy, maxChordStepPx) {
    const cc = pathArcCircumcircle(ax, ay, bx, by, cx, cy);
    if (!cc) return [[ax, ay], [bx, by]];
    const ta = Math.atan2(ay - cc.oy, ax - cc.ox);
    const tb = Math.atan2(by - cc.oy, bx - cc.ox);
    const tc = Math.atan2(cy - cc.oy, cx - cc.ox);
    const spanAB = pathArcAngleDiffCCW(ta, tb);
    let tStart, span, reverseOrder;
    if (pathArcPointBetweenAnglesCCW(ta, tc, spanAB)) {
      tStart = ta;
      span = spanAB;
      reverseOrder = false;
    } else {
      tStart = tb;
      span = pathArcAngleDiffCCW(tb, ta);
      reverseOrder = true;
    }
    const arcLen = cc.r * span;
    const mcs = Math.max(3, typeof maxChordStepPx === 'number' && maxChordStepPx > 0 ? maxChordStepPx : CELL_SIZE * 0.28);
    const n = Math.max(8, Math.min(96, Math.ceil(arcLen / mcs)));
    const pts = [];
    for (let i = 0; i <= n; i++) {
      const ang = tStart + (span * i) / n;
      pts.push([cc.ox + cc.r * Math.cos(ang), cc.oy + cc.r * Math.sin(ang)]);
    }
    if (reverseOrder) pts.reverse();
    pts[0] = [ax, ay];
    pts[pts.length - 1] = [bx, by];
    return pts;
  }
  /** Subdivide polyline so each segment length ≤ maxStepPx (smoother grid snap for arcs). */
  function pathArcDensifyPolylinePx(pts, maxStepPx) {
    if (!pts || pts.length < 2) return pts ? pts.slice() : [];
    const m = Math.max(1e-6, maxStepPx);
    const out = [[pts[0][0], pts[0][1]]];
    for (let i = 0; i < pts.length - 1; i++) {
      const x0 = pts[i][0], y0 = pts[i][1], x1 = pts[i + 1][0], y1 = pts[i + 1][1];
      const len = Math.hypot(x1 - x0, y1 - y0);
      const steps = Math.max(1, Math.ceil(len / m));
      for (let s = 1; s <= steps; s++) {
        const t = s / steps;
        out.push([x0 + (x1 - x0) * t, y0 + (y1 - y0) * t]);
      }
    }
    return out;
  }
  function pathArcComputePreviewWorldPxFromAB(Ax, Ay, Bx, By, wx, wy) {
    const dx = Bx - Ax, dy = By - Ay;
    const chordLen = Math.hypot(dx, dy) || 1;
    const ex = dx / chordLen, ey = dy / chordLen;
    const nx = -ey, ny = ex;
    const M = [(Ax + Bx) * 0.5, (Ay + By) * 0.5];
    let h = (wx - M[0]) * nx + (wy - M[1]) * ny;
    const maxH = chordLen * PATH_ARC_MAX_BULGE_FRAC;
    h = Math.max(-maxH, Math.min(maxH, h));
    if (Math.abs(h) < PATH_ARC_MIN_BULGE_PX) return [[Ax, Ay], [Bx, By]];
    const Cx = M[0] + nx * h, Cy = M[1] + ny * h;
    return pathArcSampleThreePointWorldPx(Ax, Ay, Bx, By, Cx, Cy, Math.max(CELL_SIZE * 0.28, chordLen / 28));
  }
  function pathArcComputePreviewWorldPx(tw, vertexIndex, wx, wy) {
    if (!tw || tw.pathType === 'runway') return null;
    const verts = tw.vertices;
    if (!verts || vertexIndex <= 0 || vertexIndex >= verts.length - 1) return null;
    const A = cellToPixel(verts[vertexIndex - 1].col, verts[vertexIndex - 1].row);
    const B = cellToPixel(verts[vertexIndex + 1].col, verts[vertexIndex + 1].row);
    return pathArcComputePreviewWorldPxFromAB(A[0], A[1], B[0], B[1], wx, wy);
  }
  function pathArcCommitIslandVertexFromPreview(mk, vertexIndex, previewPx, snapToGrid) {
    if (!mk || !isLayoutPolygonMarkerKind(mk.kind) || !previewPx || previewPx.length < 2) return;
    const verts = mk.points;
    const n = verts ? verts.length : 0;
    if (!verts || n < 3 || vertexIndex < 0 || vertexIndex >= n) return;
    const prev = verts[(vertexIndex - 1 + n) % n];
    const next = verts[(vertexIndex + 1) % n];
    const Apx = [Number(prev.x), Number(prev.y)];
    const Bpx = [Number(next.x), Number(next.y)];
    const workPx = previewPx.map(function(p, j) {
      if (j === 0) return [Apx[0], Apx[1]];
      if (j === previewPx.length - 1) return [Bpx[0], Bpx[1]];
      return [p[0], p[1]];
    });
    const cells = [];
    if (previewPx.length > 2) {
      const densePx = pathArcDensifyPolylinePx(workPx, Math.max(3, CELL_SIZE * 0.11));
      for (let k = 0; k < densePx.length; k++) {
        const snap = worldPointToPixel(densePx[k][0], densePx[k][1], snapToGrid);
        const c = { x: snap[0], y: snap[1] };
        if (cells.length && cells[cells.length - 1].x === c.x && cells[cells.length - 1].y === c.y) continue;
        if (c.x === prev.x && c.y === prev.y) continue;
        cells.push(c);
      }
      while (cells.length && cells[cells.length - 1].x === next.x && cells[cells.length - 1].y === next.y) cells.pop();
    }
    if (!cells.length) {
      const M = [(Apx[0] + Bpx[0]) * 0.5, (Apx[1] + Bpx[1]) * 0.5];
      const snap = worldPointToPixel(M[0], M[1], snapToGrid);
      cells.push({ x: snap[0], y: snap[1] });
    }
    const newArr = verts.slice();
    newArr.splice(vertexIndex, 1, ...cells);
    mk.points = newArr;
    const midSel = vertexIndex + Math.max(0, Math.floor((cells.length - 1) / 2));
    state.selectedVertex = { type: 'layoutMarkerHandle', id: mk.id, handle: 'islandVertex', vertexIndex: midSel };
  }
  function pathArcCommitFromPreview(tw, vertexIndex, previewPx, snapToGrid) {
    if (!tw || tw.pathType === 'runway') return;
    if (!previewPx || previewPx.length < 2) return;
    const verts = tw.vertices;
    if (!verts || vertexIndex <= 0 || vertexIndex >= verts.length - 1) return;
    const prev = verts[vertexIndex - 1], next = verts[vertexIndex + 1];
    const Apx = cellToPixel(prev.col, prev.row);
    const Bpx = cellToPixel(next.col, next.row);
    const workPx = previewPx.map(function(p, j) {
      if (j === 0) return [Apx[0], Apx[1]];
      if (j === previewPx.length - 1) return [Bpx[0], Bpx[1]];
      return [p[0], p[1]];
    });
    const cells = [];
    if (previewPx.length > 2) {
      const densePx = pathArcDensifyPolylinePx(workPx, Math.max(3, CELL_SIZE * 0.11));
      for (let k = 0; k < densePx.length; k++) {
        const c = worldPointToCellPoint(densePx[k][0], densePx[k][1], snapToGrid);
        if (cells.length && cells[cells.length - 1].col === c.col && cells[cells.length - 1].row === c.row) continue;
        if (c.col === prev.col && c.row === prev.row) continue;
        cells.push(c);
      }
      while (cells.length && cells[cells.length - 1].col === next.col && cells[cells.length - 1].row === next.row) cells.pop();
    }
    if (!cells.length) {
      const M = [(Apx[0] + Bpx[0]) * 0.5, (Apx[1] + Bpx[1]) * 0.5];
      cells.push(worldPointToCellPoint(M[0], M[1], snapToGrid));
    }
    tw.vertices.splice(vertexIndex, 1, ...cells);
    if (typeof syncStartEndFromVertices === 'function') syncStartEndFromVertices(tw);
    const midSel = vertexIndex + Math.max(0, Math.floor((cells.length - 1) / 2));
    state.selectedVertex = { type: 'taxiway', id: tw.id, index: midSel };
    bumpPathPolylineCacheRev();
  }
  function apronLinkPolyVertexIndexForMid(lk, midIndex) {
    const mids = Array.isArray(lk.midVertices) ? lk.midVertices : [];
    const n = mids.length + 2;
    if (n < 3 || midIndex < 0 || midIndex >= mids.length) return -1;
    const standFirst = String(lk.apronDrawFirstEndpoint || 'stand') === 'stand';
    return standFirst ? (midIndex + 1) : (n - 2 - midIndex);
  }
  function pathArcCommitApronLinkFromPreview(lk, polyVertexIndex, previewPx, snapToGrid) {
    if (!lk || !previewPx || previewPx.length < 2) return;
    const poly = getApronLinkPolylineWorldPts(lk);
    const n = poly.length;
    if (n < 3) return;
    const vi = polyVertexIndex;
    if (vi <= 0 || vi >= n - 1) return;
    const Apx = poly[vi - 1], Bpx = poly[vi + 1];
    const workPx = previewPx.map(function(p, j) {
      if (j === 0) return [Apx[0], Apx[1]];
      if (j === previewPx.length - 1) return [Bpx[0], Bpx[1]];
      return [p[0], p[1]];
    });
    const cells = [];
    const prevCell = worldPointToCellPoint(Apx[0], Apx[1], snapToGrid);
    const nextCell = worldPointToCellPoint(Bpx[0], Bpx[1], snapToGrid);
    if (previewPx.length > 2) {
      const densePx = pathArcDensifyPolylinePx(workPx, Math.max(3, CELL_SIZE * 0.11));
      for (let k = 0; k < densePx.length; k++) {
        const c = worldPointToCellPoint(densePx[k][0], densePx[k][1], snapToGrid);
        if (cells.length && cells[cells.length - 1].col === c.col && cells[cells.length - 1].row === c.row) continue;
        if (c.col === prevCell.col && c.row === prevCell.row) continue;
        cells.push(c);
      }
      while (cells.length && cells[cells.length - 1].col === nextCell.col && cells[cells.length - 1].row === nextCell.row) cells.pop();
    }
    if (!cells.length) {
      const M = [(Apx[0] + Bpx[0]) * 0.5, (Apx[1] + Bpx[1]) * 0.5];
      cells.push(worldPointToCellPoint(M[0], M[1], snapToGrid));
    }
    const standFirst = String(lk.apronDrawFirstEndpoint || 'stand') === 'stand';
    const mids = Array.isArray(lk.midVertices) ? lk.midVertices : [];
    let midIndex = standFirst ? (vi - 1) : (mids.length - vi);
    if (midIndex < 0 || midIndex >= mids.length) return;
    if (!Array.isArray(lk.midVertices)) lk.midVertices = [];
    lk.midVertices.splice(midIndex, 1, ...cells);
    const midSel = midIndex + Math.max(0, Math.floor((cells.length - 1) / 2));
    state.selectedVertex = { type: 'apronLink', id: lk.id, kind: 'mid', midIndex: midSel };
    bumpPathPolylineCacheRev();
    markApronLinkJunctionOverlayDirty(lk.id);
  }
  function isPathArcHudVertexSelection() {
    const so = state.selectedObject;
    const sv = state.selectedVertex;
    if (!so || !so.obj) return null;
    if (so.type === 'taxiway') {
      if (!sv || sv.type !== 'taxiway' || sv.id !== so.id) return null;
      const tw = so.obj;
      if (!tw || tw.pathType === 'runway') return null;
      const idx = sv.index;
      const verts = tw.vertices;
      if (!verts || idx <= 0 || idx >= verts.length - 1) return null;
      return { kind: 'taxiway', tw: tw, idx: idx };
    }
    if (so.type === 'layoutMarker' && isLayoutPolygonMarkerKind(so.obj.kind)) {
      if (!sv || sv.type !== 'layoutMarkerHandle' || sv.handle !== 'islandVertex' || String(sv.id) !== String(so.id)) return null;
      const mk = so.obj;
      const pts = mk.points;
      const n = (pts && pts.length) || 0;
      const idx = sv.vertexIndex;
      if (n < 3 || typeof idx !== 'number' || idx < 0 || idx >= n) return null;
      return { kind: 'island', mk: mk, idx: idx };
    }
    if (so.type === 'apronLink') {
      const lk = so.obj;
      if (!lk || !sv || sv.type !== 'apronLink' || sv.id !== so.id || sv.kind !== 'mid' || typeof sv.midIndex !== 'number') return null;
      const vi = apronLinkPolyVertexIndexForMid(lk, sv.midIndex);
      const poly = getApronLinkPolylineWorldPts(lk);
      if (vi <= 0 || vi >= poly.length - 1) return null;
      return { kind: 'apronLink', lk: lk, polyVertexIndex: vi };
    }
    return null;
  }
  function clearPathArcIfStale() {
    if (!state.pathArcDrag) return;
    const d = state.pathArcDrag;
    if (d.islandMarkerId != null) {
      const mk = (state.layoutMarkers || []).find(function(m) { return m && String(m.id) === String(d.islandMarkerId); });
      const n = (mk && isLayoutPolygonMarkerKind(mk.kind) && mk.points && mk.points.length) || 0;
      if (!mk || !isLayoutPolygonMarkerKind(mk.kind) || d.vertexIndex < 0 || d.vertexIndex >= n) state.pathArcDrag = null;
      return;
    }
    if (d.apronLinkId != null) {
      const lk = (state.apronLinks || []).find(function(l) { return l && l.id === d.apronLinkId; });
      const poly = lk ? getApronLinkPolylineWorldPts(lk) : [];
      if (!lk || poly.length < 3 || d.polyVertexIndex <= 0 || d.polyVertexIndex >= poly.length - 1) state.pathArcDrag = null;
      return;
    }
    const tw = state.taxiways.find(function(t) { return t.id === d.taxiwayId; });
    if (!tw || tw.pathType === 'runway' || d.vertexIndex <= 0 || d.vertexIndex >= (tw.vertices || []).length - 1) state.pathArcDrag = null;
  }
  function insertSelectedVertexAt(wx, wy, snapToGrid) {
    if (!state.selectedObject || !state.selectedObject.obj) return false;
    const sel = state.selectedObject;
    if (sel.type === 'terminal') {
      const term = sel.obj;
      const hit = findInsertSegment(term.vertices, !!term.closed, wx, wy);
      if (!hit) return false;
      const pt = worldPointToCellPoint(hit.near[0], hit.near[1], snapToGrid);
      pushUndo();
      term.vertices.splice(hit.insertIndex, 0, pt);
      state.selectedVertex = { type: 'terminal', id: term.id, index: hit.insertIndex };
      updateObjectInfo();
      draw();
      return true;
    }
    if (sel.type === 'taxiway') {
      const tw = sel.obj;
      const hit = findInsertSegment(tw.vertices, false, wx, wy);
      if (!hit) return false;
      const pt = worldPointToCellPoint(hit.near[0], hit.near[1], snapToGrid);
      pushUndo();
      tw.vertices.splice(hit.insertIndex, 0, pt);
      if (typeof syncStartEndFromVertices === 'function') syncStartEndFromVertices(tw);
      state.selectedVertex = { type: 'taxiway', id: tw.id, index: hit.insertIndex };
      if (typeof redrawLayoutAfterEdit === 'function') redrawLayoutAfterEdit();
      else if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths(); else draw();
      return true;
    }
    if (sel.type === 'apronLink') {
      const lk = sel.obj;
      const mids = (Array.isArray(lk.midVertices) ? lk.midVertices.slice() : []);
      const midsPx = mids.map(function(v) {
        if (v && isFinite(Number(v.x)) && isFinite(Number(v.y))) return [Number(v.x), Number(v.y)];
        return cellToPixel(Number(v.col), Number(v.row));
      });
      const poly = [getApronLinkStandEndPx(lk)].concat(midsPx).concat([[Number(lk.tx), Number(lk.ty)]]);
      const hit = findInsertSegment(poly, false, wx, wy);
      if (!hit) return false;
      const pt = worldPointToCellPoint(hit.near[0], hit.near[1], snapToGrid);
      pushUndo();
      if (!Array.isArray(lk.midVertices)) lk.midVertices = [];
      lk.midVertices.splice(Math.max(0, hit.insertIndex - 1), 0, pt);
      state.selectedVertex = { type: 'apronLink', id: lk.id, kind: 'mid', midIndex: Math.max(0, hit.insertIndex - 1) };
      markApronLinkJunctionOverlayDirty(lk.id);
      if (typeof redrawLayoutAfterEdit === 'function') redrawLayoutAfterEdit();
      else if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths(); else draw();
      return true;
    }
    if (sel.type === 'layoutMarker') {
      const mk = sel.obj;
      if (!mk || !isLayoutPolygonMarkerKind(mk.kind) || !Array.isArray(mk.points) || mk.points.length < 2) return false;
      const hit = findInsertSegment(mk.points, true, wx, wy);
      if (!hit) return false;
      const pt = worldPointToPixel(hit.near[0], hit.near[1], snapToGrid);
      pushUndo();
      mk.points.splice(hit.insertIndex, 0, { x: pt[0], y: pt[1] });
      state.selectedVertex = { type: 'layoutMarkerHandle', id: mk.id, handle: 'islandVertex', vertexIndex: hit.insertIndex };
      if (typeof updateObjectInfo === 'function') updateObjectInfo();
      draw();
      return true;
    }
    return false;
  }

  function snapWorldPointToTaxiwayPolyline(wx, wy, taxiwayId) {
    const tw = (state.taxiways || []).find(t => t.id === taxiwayId);
    if (!tw || !tw.vertices || tw.vertices.length < 2) return null;
    const click = [wx, wy];
    let best = null;
    let bestD2 = Infinity;
    for (let i = 0; i < tw.vertices.length - 1; i++) {
      const [x1, y1] = cellToPixel(tw.vertices[i].col, tw.vertices[i].row);
      const [x2, y2] = cellToPixel(tw.vertices[i + 1].col, tw.vertices[i + 1].row);
      const near = closestPointOnSegment([x1, y1], [x2, y2], click);
      if (!near) continue;
      const d2 = dist2(near, click);
      if (d2 < bestD2) { bestD2 = d2; best = near; }
    }
    return best;
  }
  function snapWorldPointToLayoutObjectsForMarker(wx, wy) {
    const click = [wx, wy];
    let bestD2 = Infinity;
    let best = null;
    function considerPoint(pt) {
      if (!pt || pt.length < 2) return;
      const d2 = dist2(pt, click);
      if (d2 < bestD2) {
        bestD2 = d2;
        best = [pt[0], pt[1]];
      }
    }
    function considerSeg(p1, p2) {
      const near = closestPointOnSegment(p1, p2, click);
      if (!near) return;
      considerPoint(near);
    }
    (state.terminals || []).forEach(function(t) {
      if (!t || !t.vertices || t.vertices.length < 2) return;
      const verts = t.vertices;
      const n = verts.length;
      const segCount = t.closed ? n : (n - 1);
      for (let i = 0; i < segCount; i++) {
        const j = t.closed ? ((i + 1) % n) : (i + 1);
        considerSeg(cellToPixel(verts[i].col, verts[i].row), cellToPixel(verts[j].col, verts[j].row));
      }
    });
    (state.taxiways || []).forEach(function(tw) {
      if (!tw) return;
      const poly = typeof getOrderedPoints === 'function' ? getOrderedPoints(tw) : getTaxiwayOrderedPoints(tw);
      if (!poly || poly.length < 2) return;
      for (let i = 0; i < poly.length - 1; i++) considerSeg(poly[i], poly[i + 1]);
    });
    (state.holdingPoints || []).forEach(function(hp) {
      if (!hp || !isFinite(hp.x) || !isFinite(hp.y)) return;
      considerPoint([hp.x, hp.y]);
    });
    (state.apronLinks || []).forEach(function(lk) {
      const poly = typeof getApronLinkPolylineWorldPts === 'function' ? getApronLinkPolylineWorldPts(lk) : null;
      if (!poly || poly.length < 2) return;
      for (let i = 0; i < poly.length - 1; i++) considerSeg(poly[i], poly[i + 1]);
    });
    (state.pbbStands || []).forEach(function(pbb) {
      const corners = typeof getPBBStandCorners === 'function' ? getPBBStandCorners(pbb) : null;
      if (!corners || corners.length < 2) return;
      const m = corners.length;
      for (let i = 0; i < m; i++) considerSeg(corners[i], corners[(i + 1) % m]);
    });
    (state.remoteStands || []).forEach(function(st) {
      const corners = typeof getRemoteStandCorners === 'function' ? getRemoteStandCorners(st) : null;
      if (!corners || corners.length < 2) return;
      const m = corners.length;
      for (let i = 0; i < m; i++) considerSeg(corners[i], corners[(i + 1) % m]);
    });
    (state.tempStands || []).forEach(function(st) {
      const corners = typeof getRemoteStandCorners === 'function' ? getRemoteStandCorners(st) : null;
      if (!corners || corners.length < 2) return;
      const m = corners.length;
      for (let i = 0; i < m; i++) considerSeg(corners[i], corners[(i + 1) % m]);
    });
    return best == null ? null : { pt: best, d2: bestD2 };
  }
  function markerAreaSnapWorldToPlacementPx(wx, wy, snapToGrid) {
    const click = [wx, wy];
    const gridPx = worldPointToPixel(wx, wy, snapToGrid);
    const maxD2 = Math.pow(CELL_SIZE * HIT_TW_SEG_CF, 2);
    const pack = snapWorldPointToLayoutObjectsForMarker(wx, wy);
    if (pack && pack.d2 <= maxD2 && pack.d2 <= dist2(gridPx, click)) return pack.pt;
    return gridPx;
  }

  function hitTestApronLinkVertex(wx, wy) {
    if (!state.selectedObject || state.selectedObject.type !== 'apronLink') return null;
    const lk = state.selectedObject.obj;
    if (!lk || lk.id !== state.selectedObject.id) return null;
    const click = [wx, wy];
    const maxD2 = (CELL_SIZE * HIT_TW_VTX_CF) ** 2;
    let best = null;
    let bestD2 = maxD2;
    const tx = Number(lk.tx), ty = Number(lk.ty);
    if (isFinite(tx) && isFinite(ty)) {
      const d2 = dist2([tx, ty], click);
      if (d2 < bestD2) { bestD2 = d2; best = { linkId: lk.id, kind: 'taxiway' }; }
    }
    (lk.midVertices || []).forEach((v, idx) => {
      const px = v && isFinite(Number(v.x)) && isFinite(Number(v.y))
        ? [Number(v.x), Number(v.y)]
        : cellToPixel(Number(v.col), Number(v.row));
      const d2 = dist2(px, click);
      if (d2 < bestD2) { bestD2 = d2; best = { linkId: lk.id, kind: 'mid', midIndex: idx }; }
    });
    return best;
  }

  function isSelectedVertex(type, objectId, index) {
    const sv = state.selectedVertex;
    return !!(sv && sv.type === type && sv.id === objectId && sv.index === index);
  }

  function removeSelectedVertex() {
    const sv = state.selectedVertex;
    if (!sv) return false;
    if (sv.type === 'terminal') {
      const term = state.terminals.find(t => t.id === sv.id);
      if (!term || !Array.isArray(term.vertices) || sv.index < 0 || sv.index >= term.vertices.length) return false;
      if (term.closed && term.vertices.length <= 3) return false;
      pushUndo();
      term.vertices.splice(sv.index, 1);
      if (term.vertices.length < 3) term.closed = false;
      state.selectedVertex = null;
      if (state.currentTerminalId === term.id) syncPanelFromState();
      updateObjectInfo();
      draw();
      return true;
    }
    if (sv.type === 'taxiway') {
      const tw = state.taxiways.find(t => t.id === sv.id);
      if (!tw || !Array.isArray(tw.vertices) || sv.index < 0 || sv.index >= tw.vertices.length) return false;
      if (tw.vertices.length <= 2) return false;
      pushUndo();
      tw.vertices.splice(sv.index, 1);
      if (typeof syncStartEndFromVertices === 'function' && tw.vertices.length >= 2) syncStartEndFromVertices(tw);
      state.selectedVertex = null;
      syncPanelFromState();
      updateObjectInfo();
      if (typeof redrawLayoutAfterEdit === 'function') redrawLayoutAfterEdit();
      else if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths(); else draw();
      return true;
    }
    if (sv.type === 'apronLink') {
      if (sv.kind !== 'mid') return false;
      const lk = state.apronLinks.find(l => l.id === sv.id);
      if (!lk || !Array.isArray(lk.midVertices) || sv.midIndex < 0 || sv.midIndex >= lk.midVertices.length) return false;
      pushUndo();
      lk.midVertices.splice(sv.midIndex, 1);
      if (!lk.midVertices.length) delete lk.midVertices;
      state.selectedVertex = null;
      updateObjectInfo();
      markApronLinkJunctionOverlayDirty(lk.id);
      if (typeof redrawLayoutAfterEdit === 'function') redrawLayoutAfterEdit();
      else if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths(); else draw();
      return true;
    }
    if (sv.type === 'layoutMarkerHandle' && sv.handle === 'islandVertex') {
      const mk = (state.layoutMarkers || []).find(function(m) { return m && String(m.id) === String(sv.id); });
      if (!mk || !isLayoutPolygonMarkerKind(mk.kind) || !Array.isArray(mk.points)) return false;
      const idx = sv.vertexIndex;
      if (typeof idx !== 'number' || idx < 0 || idx >= mk.points.length || mk.points.length <= 3) return false;
      pushUndo();
      mk.points.splice(idx, 1);
      state.selectedVertex = null;
      if (typeof updateObjectInfo === 'function') updateObjectInfo();
      draw();
      return true;
    }
    return false;
  }

  function removeLastDrawingVertex() {
    if (state.terminalDrawingId) {
      const term = state.terminals.find(t => t.id === state.terminalDrawingId);
      if (!term || !Array.isArray(term.vertices) || !term.vertices.length) return false;
      pushUndo();
      term.vertices.pop();
      if (!term.vertices.length) state.layoutPathDrawPointer = null;
      state.selectedVertex = null;
      syncPanelFromState();
      updateObjectInfo();
      draw();
      return true;
    }
    if (state.taxiwayDrawingId) {
      const tw = state.taxiways.find(t => t.id === state.taxiwayDrawingId);
      if (!tw || !Array.isArray(tw.vertices) || !tw.vertices.length) return false;
      pushUndo();
      tw.vertices.pop();
      if (!tw.vertices.length) state.layoutPathDrawPointer = null;
      if (typeof syncStartEndFromVertices === 'function' && tw.vertices.length >= 2) syncStartEndFromVertices(tw);
      else {
        tw.start_point = null;
        tw.end_point = null;
      }
      state.selectedVertex = null;
      syncPanelFromState();
      updateObjectInfo();
      if (typeof redrawLayoutAfterEdit === 'function') redrawLayoutAfterEdit();
      else if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths(); else draw();
      return true;
    }
    if (settingModeSelect.value === 'apronTaxiway' && state.apronLinkDrawing && state.apronLinkTemp) {
      if (state.apronLinkMidpoints && state.apronLinkMidpoints.length) {
        state.apronLinkMidpoints.pop();
        draw();
        return true;
      }
      state.apronLinkTemp = null;
      state.apronLinkMidpoints = [];
      state.apronLinkPointerWorld = null;
      draw();
      return true;
    }
    if (settingModeSelect.value === 'marker' && state.markerDrawing && getMarkerSubKindFromPanel() === 'island' && state.markerIslandDraft && state.markerIslandDraft.points && state.markerIslandDraft.points.length) {
      state.markerIslandDraft.points.pop();
      if (!state.markerIslandDraft.points.length) state.markerIslandDraft = null;
      state.markerIslandHoverWorld = null;
      draw();
      return true;
    }
    if (settingModeSelect.value === 'marker' && state.markerDrawing && getMarkerSubKindFromPanel() === 'area' && state.markerAreaDraft && state.markerAreaDraft.points && state.markerAreaDraft.points.length) {
      state.markerAreaDraft.points.pop();
      if (!state.markerAreaDraft.points.length) state.markerAreaDraft = null;
      state.markerAreaHoverWorld = null;
      draw();
      return true;
    }
    return false;
  }

  function getCurrentTerminal() {
    if (state.selectedObject && state.selectedObject.type === 'terminal' && state.selectedObject.obj) {


      return state.selectedObject.obj;
    }
    if (state.currentTerminalId) {
      const t = state.terminals.find(x => x.id === state.currentTerminalId);
      if (t) return t;
    }
    return null;
  }

  function polygonAreaM2(vertices) {
    if (!vertices || vertices.length < 3) return 0;
    let area = 0;
    const n = vertices.length;
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n;
      area += vertices[i].col * vertices[j].row;
      area -= vertices[j].col * vertices[i].row;
    }
    return Math.abs(area) * 0.5 * CELL_SIZE * CELL_SIZE;
  }

  function syncPanelFromState() {
    document.getElementById('gridCellSize').value = CELL_SIZE;
    document.getElementById('gridCols').value = GRID_COLS;
    document.getElementById('gridRows').value = GRID_ROWS;
    const gridImageOpacityEl = document.getElementById('gridLayoutImageOpacity');
    const gridImageWidthEl = document.getElementById('gridLayoutImageWidthM');
    const gridImageHeightEl = document.getElementById('gridLayoutImageHeightM');
    const gridImageColEl = document.getElementById('gridLayoutImageCol');
    const gridImageRowEl = document.getElementById('gridLayoutImageRow');
    const gridImageMetaEl = document.getElementById('gridLayoutImageMeta');
    const gridImageClearBtn = document.getElementById('btnClearGridLayoutImage');
    const gridImageFileEl = document.getElementById('gridLayoutImageFile');
    const overlay = state.layoutImageOverlay;
    if (gridImageOpacityEl) gridImageOpacityEl.value = overlay ? String(overlay.opacity) : String(GRID_LAYOUT_IMAGE_DEFAULTS.opacity);
    if (gridImageWidthEl) gridImageWidthEl.value = overlay ? String(overlay.widthM) : String(GRID_LAYOUT_IMAGE_DEFAULTS.widthM);
    if (gridImageHeightEl) gridImageHeightEl.value = overlay ? String(overlay.heightM) : String(GRID_LAYOUT_IMAGE_DEFAULTS.heightM);
    if (gridImageColEl) gridImageColEl.value = overlay ? String(overlay.topLeftCol) : String(GRID_LAYOUT_IMAGE_DEFAULTS.topLeftCol);
    if (gridImageRowEl) gridImageRowEl.value = overlay ? String(overlay.topLeftRow) : String(GRID_LAYOUT_IMAGE_DEFAULTS.topLeftRow);
    if (gridImageMetaEl) gridImageMetaEl.textContent = overlay ? ('Loaded : ' + (overlay.name || 'Layout image')) : 'No file selected';
    if (gridImageClearBtn) gridImageClearBtn.disabled = !overlay;
    if (!overlay && gridImageFileEl) gridImageFileEl.value = '';
    if (state.currentTerminalId && !state.terminals.some(t => t.id === state.currentTerminalId))
      state.currentTerminalId = null;
    const term = getCurrentTerminal();
    if (term) {
      const buildingTypeSel = document.getElementById('buildingType');
      if (buildingTypeSel) {
        buildingTypeSel.innerHTML = getBuildingTypeOptionsHtml(term.buildingType);
        buildingTypeSel.value = normalizeBuildingType(term.buildingType);
      }
      document.getElementById('terminalName').value = term.name || getDefaultBuildingNameForType(term.buildingType, term.id);
      const floors = term.floors != null ? Math.max(1, parseInt(term.floors, 10) || 1) : 1;
      const f2fRaw = term.floorToFloor != null ? Number(term.floorToFloor) : (term.floorHeight != null ? Number(term.floorHeight) : 4);
      const f2f = Math.max(0.5, f2fRaw || 4);
      const totalH = term.floorHeight != null ? Number(term.floorHeight) || (floors * f2f) : (floors * f2f);
      term.floors = floors;
      term.floorToFloor = f2f;
      term.floorHeight = totalH;
      const floorsInput = document.getElementById('terminalFloors');
      const f2fInput = document.getElementById('terminalFloorToFloor');
      const totalInput = document.getElementById('terminalFloorHeight');
      if (floorsInput) floorsInput.value = floors;
      if (f2fInput) f2fInput.value = f2f;
      if (totalInput) totalInput.value = totalH;
      document.getElementById('terminalDepartureCapacity').value = term.departureCapacity != null ? term.departureCapacity : 0;
      document.getElementById('terminalArrivalCapacity').value = term.arrivalCapacity != null ? term.arrivalCapacity : 0;
    }
    syncDrawToggleButton('btnTerminalDraw', !!state.terminalDrawingId);
    if (state.selectedObject && state.selectedObject.type === 'pbb') {
      const pbb = state.selectedObject.obj;
      const nameInput = document.getElementById('standName');
      const lenInput = document.getElementById('pbbLength');
      const angleInput = document.getElementById('standAngle');
      const pbbCountInput = document.getElementById('pbbBridgeCount');
      if (nameInput) nameInput.value = pbb.name || '';
      applyIcaoCategoriesToHost('standIcaoCategories', normalizeAllowedIcaoCategories(pbb.allowedIcaoCategories));
      if (lenInput) {
        let arm = Number(pbb.pbbArmLenM);
        if (!isFinite(arm) || arm <= 0) {
          const br0 = pbb.pbbBridges && pbb.pbbBridges[0];
          const p1 = br0 && br0.points && br0.points[1], p2 = br0 && br0.points && br0.points[2];
          if (p1 && p2) arm = Math.hypot(Number(p2.x) - Number(p1.x), Number(p2.y) - Number(p1.y));
          else arm = 15;
        }
        lenInput.value = String(Math.max(1, Math.round(arm)));
      }
      if (angleInput) angleInput.value = String(Math.round(getPbbAngleDeg(pbb)));
      if (pbbCountInput) pbbCountInput.value = String(Math.max(1, parseInt(pbb.pbbCount, 10) || 1));
      const boardingWInput = document.getElementById('pbbBoardingWidth');
      const boardingHInput = document.getElementById('pbbBoardingHeight');
      if (boardingWInput) boardingWInput.value = String(getPbbBoardingWidthM(pbb));
      if (boardingHInput) boardingHInput.value = String(getPbbBoardingHeightM(pbb));
      syncStandConstraintVisibility('stand');
      renderAircraftConstraintChoices('standAircraftAccess', getStandAllowedAircraftTypes(pbb), pbb.allowedIcaoCategories);
    }
    if (state.selectedObject && state.selectedObject.type === 'remote') {
      const st = state.selectedObject.obj;
      const nameInput = document.getElementById('remoteName');
      if (nameInput) nameInput.value = st.name || '';
      applyIcaoCategoriesToHost('remoteIcaoCategories', normalizeAllowedIcaoCategories(st.allowedIcaoCategories));
      syncStandConstraintVisibility('remote');
      renderAircraftConstraintChoices('remoteAircraftAccess', getStandAllowedAircraftTypes(st), st.allowedIcaoCategories);
      renderRemoteTerminalAccessChoices(Array.isArray(st.allowedTerminals) ? st.allowedTerminals : []);
    }
    if (state.selectedObject && state.selectedObject.type === 'tempStand') {
      const st = state.selectedObject.obj;
      const nameInput = document.getElementById('tempStandName');
      if (nameInput) nameInput.value = st.name || '';
      applyIcaoCategoriesToHost('tempStandIcaoCategories', normalizeAllowedIcaoCategories(st.allowedIcaoCategories));
      syncStandConstraintVisibility('tempStand');
      renderAircraftConstraintChoices('tempStandAircraftAccess', getStandAllowedAircraftTypes(st), st.allowedIcaoCategories);
      renderTempStandTerminalAccessChoices(Array.isArray(st.allowedTerminals) ? st.allowedTerminals : []);
    }
    if (state.selectedObject && state.selectedObject.type === 'holdingPoint') {
      const hp = state.selectedObject.obj;
      const nameInput = document.getElementById('holdingPointName');
      if (nameInput) nameInput.value = hp.name || '';
    }
    if (state.selectedObject && state.selectedObject.type === 'taxiway') {
      const tw = state.selectedObject.obj;
      const nameInput = document.getElementById('taxiwayName');
      const widthInput = document.getElementById('taxiwayWidth');
      const maxExitInput = document.getElementById('taxiwayMaxExitVel');
      const minExitInput = document.getElementById('taxiwayMinExitVel');
      if (nameInput) nameInput.value = tw.name || '';
      const widthDefault = tw.pathType === 'runway'
        ? RUNWAY_PATH_DEFAULT_WIDTH
        : (tw.pathType === 'runway_exit' ? RUNWAY_EXIT_DEFAULT_WIDTH : TAXIWAY_DEFAULT_WIDTH);
      if (widthInput) widthInput.value = tw.width != null ? tw.width : widthDefault;
      const avgVelInput = document.getElementById('taxiwayAvgMoveVelocity');
      if (avgVelInput) avgVelInput.value = (tw.avgMoveVelocity != null ? tw.avgMoveVelocity : 10);
      syncPathFieldVisibilityForPathType(tw.pathType || 'taxiway');
      const runwayMinArrInput = document.getElementById('runwayMinArrVelocity');
      if (runwayMinArrInput) {
        const mav = (typeof tw.minArrVelocity === 'number' && isFinite(tw.minArrVelocity) && tw.minArrVelocity > 0)
          ? Math.max(1, Math.min(150, tw.minArrVelocity))
          : 15;
        runwayMinArrInput.value = mav;
      }
      const runwayLineupInputCw = document.getElementById('runwayLineupDistM_CW');
      const runwayLineupInputCcw = document.getElementById('runwayLineupDistM_CCW');
      if (tw.pathType === 'runway') {
        if (runwayLineupInputCw) runwayLineupInputCw.value = String(getRunwayLineupDistMByDirection(tw, 'clockwise'));
        if (runwayLineupInputCcw) runwayLineupInputCcw.value = String(getRunwayLineupDistMByDirection(tw, 'counter_clockwise'));
      }
      const runwayStartDispInput = document.getElementById('runwayStartDisplacedThresholdM');
      if (runwayStartDispInput && tw.pathType === 'runway') runwayStartDispInput.value = String(getEffectiveRunwayStartDisplacedThresholdM(tw));
      const runwayStartBlastInput = document.getElementById('runwayStartBlastPadM');
      if (runwayStartBlastInput && tw.pathType === 'runway') runwayStartBlastInput.value = String(getEffectiveRunwayStartBlastPadM(tw));
      const runwayEndDispInput = document.getElementById('runwayEndDisplacedThresholdM');
      if (runwayEndDispInput && tw.pathType === 'runway') runwayEndDispInput.value = String(getEffectiveRunwayEndDisplacedThresholdM(tw));
      const runwayEndBlastInput = document.getElementById('runwayEndBlastPadM');
      if (runwayEndBlastInput && tw.pathType === 'runway') runwayEndBlastInput.value = String(getEffectiveRunwayEndBlastPadM(tw));
      if (maxExitInput) maxExitInput.value = tw.maxExitVelocity != null ? tw.maxExitVelocity : 30;
      if (minExitInput) {
        const minVal = (typeof tw.minExitVelocity === 'number' && isFinite(tw.minExitVelocity) && tw.minExitVelocity > 0)
          ? tw.minExitVelocity
          : 15;
        minExitInput.value = minVal;
      }
      if (tw.pathType === 'runway_exit') {
        const allow = getTaxiwayAllowedRunwayDirections(tw);
        renderRunwayDirectionChoices(allow);
      } else {
        renderRunwayDirectionChoices([]);
      }
      const modeSel = document.getElementById('taxiwayDirectionMode');
      let d = getTaxiwayDirection(tw);
      if (tw.pathType === 'runway' && d === 'both') d = 'clockwise';
      if (modeSel) modeSel.value = d;
      const kindSel = document.getElementById('taxiwayPathTypeKind');
      if (kindSel) {
        const ptk = tw.pathType || 'taxiway';
        if (ptk === 'general_queue_taxiway') kindSel.value = 'queue';
        else if (ptk === 'runway_exit' || ptk === 'runway_taxiway') kindSel.value = (tw.queueFlow === true) ? 'queue' : 'normal';
        else kindSel.value = 'normal';
      }
      syncPathPavementRadiosToValue(pathPavementResolvedForTaxiway(tw));
    } else if (state.selectedObject && state.selectedObject.type === 'apronLink') {
      const lk = state.selectedObject.obj;
      const nameInput = document.getElementById('apronLinkName');
      if (nameInput) nameInput.value = getApronLinkDisplayName(lk);
    } else if (state.selectedObject && state.selectedObject.type === 'layoutEdge') {
      const ed = state.selectedObject.obj;
      const nameInput = document.getElementById('edgeName');
      if (nameInput) nameInput.value = getLayoutEdgeDisplayName(ed);
    } else {
      const rm = settingModeSelect ? settingModeSelect.value : '';
      if (isPathLayoutMode(rm)) {
        const ptx = pathTypeFromLayoutMode(rm);
        syncPathFieldVisibilityForPathType(ptx);
        if (ptx === 'runway_exit') {
          const allowDef = (RW_EXIT_ALLOWED_DEFAULT && RW_EXIT_ALLOWED_DEFAULT.length) ? RW_EXIT_ALLOWED_DEFAULT : ['clockwise', 'counter_clockwise'];
          renderRunwayDirectionChoices(allowDef);
        }
        const twKindIdle = document.getElementById('taxiwayPathTypeKind');
        if (twKindIdle && ptx === 'taxiway') twKindIdle.value = 'normal';
        if (twKindIdle && (ptx === 'runway_exit' || ptx === 'runway_taxiway')) twKindIdle.value = 'normal';
        syncPathPavementRadiosToValue(pathPavementDefaultForPathType(ptx));
      }
      else {
        const maxExitWrap = document.getElementById('runwayMaxExitVelWrap');
        if (maxExitWrap) maxExitWrap.style.display = 'none';
        const minExitWrap = document.getElementById('runwayMinExitVelWrap');
        if (minExitWrap) minExitWrap.style.display = 'none';
        const runwayMinArrWrap = document.getElementById('runwayMinArrVelocityWrap');
        if (runwayMinArrWrap) runwayMinArrWrap.style.display = 'none';
        const runwayLineupWrap = document.getElementById('runwayLineupDistWrap');
        if (runwayLineupWrap) runwayLineupWrap.style.display = 'none';
        const runwayStartDispWrap = document.getElementById('runwayStartDisplacedThresholdWrap');
        if (runwayStartDispWrap) runwayStartDispWrap.style.display = 'none';
        const runwayStartBlastWrap = document.getElementById('runwayStartBlastPadWrap');
        if (runwayStartBlastWrap) runwayStartBlastWrap.style.display = 'none';
        const runwayEndDispWrap = document.getElementById('runwayEndDisplacedThresholdWrap');
        if (runwayEndDispWrap) runwayEndDispWrap.style.display = 'none';
        const runwayEndBlastWrap = document.getElementById('runwayEndBlastPadWrap');
        if (runwayEndBlastWrap) runwayEndBlastWrap.style.display = 'none';
        const taxiwayAvgWrap = document.getElementById('taxiwayAvgVelocityWrap');
        if (taxiwayAvgWrap) taxiwayAvgWrap.style.display = 'none';
        const rwDirWrap = document.getElementById('runwayExitAllowedDirectionWrap');
        if (rwDirWrap) rwDirWrap.style.display = 'none';
        const taxiwayTypeWrapIdle = document.getElementById('taxiwayTypeWrap');
        if (taxiwayTypeWrapIdle) taxiwayTypeWrapIdle.style.display = 'none';
      }
      const selIsTerminal = state.selectedObject && state.selectedObject.type === 'terminal';
      if (!selIsTerminal) {
        const buildingTypeSel = document.getElementById('buildingType');
        if (buildingTypeSel) {
          buildingTypeSel.innerHTML = getBuildingTypeOptionsHtml(BUILDING_TYPE_DEFAULT);
          buildingTypeSel.value = BUILDING_TYPE_DEFAULT;
        }
        const terminalNameInput = document.getElementById('terminalName');
        if (terminalNameInput && rm === 'terminal') terminalNameInput.value = getDefaultBuildingNameForType(BUILDING_TYPE_DEFAULT, null);
      }
      const skipStandPanelIdleReset = state.selectedObject && (
        state.selectedObject.type === 'pbb' ||
        state.selectedObject.type === 'remote' ||
        state.selectedObject.type === 'tempStand'
      );
      if (!skipStandPanelIdleReset) {
        applyIcaoCategoriesToHost('standIcaoCategories', ['C']);
        syncStandConstraintVisibility('stand');
        renderAircraftConstraintChoices('standAircraftAccess', aircraftTypeIdsForIcaoLetters(['C']), ['C']);
        applyIcaoCategoriesToHost('remoteIcaoCategories', ['C']);
        syncStandConstraintVisibility('remote');
        renderAircraftConstraintChoices('remoteAircraftAccess', aircraftTypeIdsForIcaoLetters(['C']), ['C']);
        renderRemoteTerminalAccessChoices([]);
        applyIcaoCategoriesToHost('tempStandIcaoCategories', ['C']);
        syncStandConstraintVisibility('tempStand');
        renderAircraftConstraintChoices('tempStandAircraftAccess', aircraftTypeIdsForIcaoLetters(['C']), ['C']);
        renderTempStandTerminalAccessChoices([]);
      }
      const tempStandNameInput = document.getElementById('tempStandName');
      if (tempStandNameInput && rm === 'tempStand' && !(state.selectedObject && state.selectedObject.type === 'tempStand')) tempStandNameInput.value = '';
      const standNameInputIdle = document.getElementById('standName');
      if (standNameInputIdle && rm === 'pbb' && !(state.selectedObject && state.selectedObject.type === 'pbb')) standNameInputIdle.value = '';
      const remoteNameInputIdle = document.getElementById('remoteName');
      if (remoteNameInputIdle && rm === 'remote' && !(state.selectedObject && state.selectedObject.type === 'remote')) remoteNameInputIdle.value = '';
      const taxiwayNameInputIdle = document.getElementById('taxiwayName');
      if (taxiwayNameInputIdle && isPathLayoutMode(rm) && !(state.selectedObject && state.selectedObject.type === 'taxiway')) taxiwayNameInputIdle.value = '';
      const apronLinkNameInput = document.getElementById('apronLinkName');
      if (apronLinkNameInput && rm === 'apronTaxiway') apronLinkNameInput.value = '';
      const edgeNameInput = document.getElementById('edgeName');
      if (edgeNameInput && rm === 'edge') edgeNameInput.value = '';
      const holdingPointNameInput = document.getElementById('holdingPointName');
      if (holdingPointNameInput && rm === 'holdingPoint') holdingPointNameInput.value = getDefaultHoldingPointLabel();
    }
    syncDrawToggleButton('btnTaxiwayDraw', !!state.taxiwayDrawingId);
    syncDrawToggleButton('btnApronLinkDraw', !!state.apronLinkDrawing);
    syncDrawToggleButton('btnPbbDraw', !!state.pbbDrawing);
    syncDrawToggleButton('btnRemoteDraw', !!state.remoteDrawing);
    syncDrawToggleButton('btnTempStandDraw', !!state.tempStandDrawing);
    syncDrawToggleButton('btnHoldingPointDraw', !!state.holdingPointDrawing);
    syncDrawToggleButton('btnMarkerDraw', !!state.markerDrawing);
    renderObjectList();
  }

  function syncStateFromPanel() {
    var el = function(id) { return document.getElementById(id); };
    if (el('gridCellSize')) CELL_SIZE = Math.max(5, Number(el('gridCellSize').value) || 5);
    if (el('gridCols')) GRID_COLS = Math.max(5, Math.min(1000, parseInt(el('gridCols').value, 10) || 200));
    if (el('gridRows')) GRID_ROWS = Math.max(5, Math.min(1000, parseInt(el('gridRows').value, 10) || 200));
    if (state.layoutImageOverlay) {
      state.layoutImageOverlay.opacity = clampLayoutImageOpacity(el('gridLayoutImageOpacity') ? el('gridLayoutImageOpacity').value : state.layoutImageOverlay.opacity);
      state.layoutImageOverlay.widthM = clampLayoutImageSize(el('gridLayoutImageWidthM') ? el('gridLayoutImageWidthM').value : state.layoutImageOverlay.widthM, state.layoutImageOverlay.widthM);
      state.layoutImageOverlay.heightM = clampLayoutImageSize(el('gridLayoutImageHeightM') ? el('gridLayoutImageHeightM').value : state.layoutImageOverlay.heightM, state.layoutImageOverlay.heightM);
      state.layoutImageOverlay.topLeftCol = clampLayoutImagePoint(el('gridLayoutImageCol') ? el('gridLayoutImageCol').value : state.layoutImageOverlay.topLeftCol, state.layoutImageOverlay.topLeftCol);
      state.layoutImageOverlay.topLeftRow = clampLayoutImagePoint(el('gridLayoutImageRow') ? el('gridLayoutImageRow').value : state.layoutImageOverlay.topLeftRow, state.layoutImageOverlay.topLeftRow);
    }
    var t = getCurrentTerminal();
    if (t) {
      if (el('terminalName')) {
        const rawTn = (el('terminalName').value || '').trim();
        if (rawTn && findDuplicateLayoutName('terminal', t.id, rawTn)) {
          alertDuplicateLayoutName();
          el('terminalName').value = t.name || '';
        } else {
          t.name = rawTn || t.name;
        }
      }
      if (el('buildingType')) t.buildingType = normalizeBuildingType(el('buildingType').value || t.buildingType);
      if (el('terminalFloors')) t.floors = Math.max(1, parseInt(el('terminalFloors').value, 10) || 1);
      if (el('terminalFloorToFloor')) t.floorToFloor = Math.max(0.5, Number(el('terminalFloorToFloor').value) || 4);
      t.floorHeight = (t.floors || 1) * (t.floorToFloor || 4);
      if (el('terminalDepartureCapacity')) t.departureCapacity = Math.max(0, parseInt(el('terminalDepartureCapacity').value, 10) || 0);
      if (el('terminalArrivalCapacity')) t.arrivalCapacity = Math.max(0, parseInt(el('terminalArrivalCapacity').value, 10) || 0);
    }
    if (state.selectedObject && state.selectedObject.type === 'pbb') {
      var pbb = state.selectedObject.obj;
      if (el('standName')) {
        const rawSn = (el('standName').value || '').trim();
        if (rawSn && findDuplicateLayoutName('pbb', pbb.id, rawSn)) {
          alertDuplicateLayoutName();
          el('standName').value = pbb.name || '';
        } else {
          pbb.name = rawSn;
        }
      }
      applyUnifiedStandConstraintFromPanelToObject(pbb, 'standIcaoCategories', 'standAircraftAccess');
    }
    if (state.selectedObject && state.selectedObject.type === 'remote') {
      var st = state.selectedObject.obj;
      if (el('remoteName')) {
        const rawRn = (el('remoteName').value || '').trim();
        if (rawRn && findDuplicateLayoutName('remote', st.id, rawRn)) {
          alertDuplicateLayoutName();
          el('remoteName').value = st.name || '';
        } else {
          st.name = rawRn;
        }
      }
      applyUnifiedStandConstraintFromPanelToObject(st, 'remoteIcaoCategories', 'remoteAircraftAccess');
      const accWrap = document.getElementById('remoteTerminalAccess');
      if (accWrap) {
        const checks = accWrap.querySelectorAll('.remote-term-check');
        const allowed = [];
        checks.forEach(function(ch) {
          if (ch.checked) {
            const id = ch.getAttribute('data-item-id');
            if (id) allowed.push(id);
          }
        });
        st.allowedTerminals = allowed;
      }
    }
    if (state.selectedObject && state.selectedObject.type === 'tempStand') {
      var tst = state.selectedObject.obj;
      if (el('tempStandName')) {
        const rawTn = (el('tempStandName').value || '').trim();
        if (rawTn && findDuplicateLayoutName('tempStand', tst.id, rawTn)) {
          alertDuplicateLayoutName();
          el('tempStandName').value = tst.name || '';
        } else {
          tst.name = rawTn;
        }
      }
      applyUnifiedStandConstraintFromPanelToObject(tst, 'tempStandIcaoCategories', 'tempStandAircraftAccess');
      const tempAccWrap = document.getElementById('tempStandTerminalAccess');
      if (tempAccWrap) {
        const checks = tempAccWrap.querySelectorAll('.remote-term-check');
        const allowed = [];
        checks.forEach(function(ch) {
          if (ch.checked) {
            const id = ch.getAttribute('data-item-id');
            if (id) allowed.push(id);
          }
        });
        tst.allowedTerminals = allowed;
      }
    }
    if (state.selectedObject && state.selectedObject.type === 'holdingPoint') {
      var hpo = state.selectedObject.obj;
      if (el('holdingPointName')) {
        const rawHp = (el('holdingPointName').value || '').trim();
        if (rawHp && findDuplicateLayoutName('holdingPoint', hpo.id, rawHp)) {
          alertDuplicateLayoutName();
          el('holdingPointName').value = hpo.name || '';
        } else {
          hpo.name = rawHp;
        }
      }
    }
    if (state.selectedObject && state.selectedObject.type === 'taxiway') {
      var tw = state.selectedObject.obj;
      if (el('taxiwayName')) {
        const rawTw = (el('taxiwayName').value || '').trim();
        if (rawTw && findDuplicateLayoutName('taxiway', tw.id, rawTw)) {
          alertDuplicateLayoutName();
          el('taxiwayName').value = tw.name || '';
        } else {
          tw.name = rawTw;
        }
      }
      if (el('taxiwayWidth')) {
        const pathType = tw.pathType || 'taxiway';
        const fb = pathType === 'runway' ? RUNWAY_PATH_DEFAULT_WIDTH : (pathType === 'runway_exit' ? RUNWAY_EXIT_DEFAULT_WIDTH : TAXIWAY_DEFAULT_WIDTH);
        tw.width = clampTaxiwayWidthM(pathType, el('taxiwayWidth').value, fb);
      }
      if (document.getElementById('pathPavement')) {
        tw.pavement = getPathPavementFromPanelForPathType(tw.pathType || 'taxiway');
      }
      if (el('taxiwayMaxExitVel')) {
        const mv = Number(el('taxiwayMaxExitVel').value);
        if (tw.pathType === 'runway_exit') tw.maxExitVelocity = isFinite(mv) && mv > 0 ? mv : null;
        else delete tw.maxExitVelocity;
      }
      if (el('taxiwayMinExitVel') && tw.pathType === 'runway_exit') {
        const mv2 = Number(el('taxiwayMinExitVel').value);
        let v = isFinite(mv2) && mv2 > 0 ? mv2 : 15;
        if (typeof tw.maxExitVelocity === 'number' && isFinite(tw.maxExitVelocity) && v > tw.maxExitVelocity) v = tw.maxExitVelocity;
        tw.minExitVelocity = v;
        tw.allowedRwDirections = getRunwayExitAllowedDirectionsFromPanel();
      } else if (tw.pathType !== 'runway_exit') {
        delete tw.minExitVelocity;
        delete tw.allowedRwDirections;
      }
      if (el('taxiwayDirectionMode')) {
        let dirVal = el('taxiwayDirectionMode').value || '';
        if (tw.pathType === 'runway') {
          runwayReverseVerticesIfDirectionChanged(tw, dirVal);
          tw.direction = (dirVal === 'counter_clockwise') ? 'counter_clockwise' : 'clockwise';
        } else tw.direction = dirVal || 'both';
      }
      if (el('taxiwayPathTypeKind')) {
        const ptCur = tw.pathType || 'taxiway';
        if (ptCur === 'taxiway' || ptCur === 'general_queue_taxiway') {
          const kind = String(el('taxiwayPathTypeKind').value || 'normal');
          tw.pathType = (kind === 'queue') ? 'general_queue_taxiway' : 'taxiway';
          delete tw.queueFlow;
        } else if (ptCur === 'runway_exit' || ptCur === 'runway_taxiway') {
          const kindR = String(el('taxiwayPathTypeKind').value || 'normal');
