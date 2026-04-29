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
