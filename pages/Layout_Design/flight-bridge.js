    return getStandAircraftMarkerWorldPxForRemoteLike(stand);
  }
  function getStandBoundsRect(cx, cy, sizeM) {
    const h = sizeM / 2;
    return { left: cx - h, right: cx + h, top: cy - h, bottom: cy + h };
  }
  function normalizeAngleDeg(deg) {
    let a = Number(deg);
    if (!isFinite(a)) a = 0;
    while (a > 180) a -= 360;
    while (a <= -180) a += 360;
    return a;
  }
  function getRemoteStandCenterPx(st) {
    if (!st) return [0, 0];
    if (st.apronSiteX != null && st.apronSiteY != null) {
      return [Number(st.apronSiteX), Number(st.apronSiteY)];
    }
    if (typeof st.x === 'number' && isFinite(st.x) && typeof st.y === 'number' && isFinite(st.y)) {
      return [Number(st.x), Number(st.y)];
    }
    return cellToPixel(st.col || 0, st.row || 0);
  }
  /** Temp stand: taxiway centerline snap (sim_input junctionX/Y); defaults to stand x,y. */
  function getTempStandTaxiwayJunctionPx(st) {
    if (!st) return [0, 0];
    const jx = st.junctionX != null ? Number(st.junctionX) : NaN;
    const jy = st.junctionY != null ? Number(st.junctionY) : NaN;
    if (Number.isFinite(jx) && Number.isFinite(jy)) return [jx, jy];
    return getRemoteStandCenterPx(st);
  }
  function getRemoteStandAngleRad(st) {
    const deg = normalizeAngleDeg(st && st.angleDeg != null ? st.angleDeg : 0);
    return deg * Math.PI / 180;
  }
  function getRemoteStandCorners(stLike) {
    const [cx, cy] = getRemoteStandCenterPx(stLike);
    const cat = (stLike && stLike.category) || 'C';
    const dep = getStandDepthMeters(cat);
    const halfD = dep / 2;
    const halfW = getStandWidthMeters(cat) / 2;
    const angle = getRemoteStandAngleRad(stLike);
    const shiftX = standStopbarCenterShiftLocalX(dep, cat);
    const cos = Math.cos(angle), sin = Math.sin(angle);
    return [
      [cx + ((-halfD + shiftX))*cos - (-halfW)*sin, cy + ((-halfD + shiftX))*sin + (-halfW)*cos],
      [cx + (( halfD + shiftX))*cos - (-halfW)*sin, cy + (( halfD + shiftX))*sin + (-halfW)*cos],
      [cx + (( halfD + shiftX))*cos - ( halfW)*sin, cy + (( halfD + shiftX))*sin + ( halfW)*cos],
      [cx + ((-halfD + shiftX))*cos - ( halfW)*sin, cy + ((-halfD + shiftX))*sin + ( halfW)*cos]
    ];
  }
  function rectsOverlap(a, b) {
    return !(a.right <= b.left || a.left >= b.right || a.bottom <= b.top || a.top >= b.bottom);
  }
  function getPbbAnchorPx(pbb) {
    const x1 = Number(pbb && pbb.x1);
    const y1 = Number(pbb && pbb.y1);
    if (Number.isFinite(x1) && Number.isFinite(y1)) return [x1, y1];
    const bridges = Array.isArray(pbb && pbb.pbbBridges) ? pbb.pbbBridges : [];
    const starts = bridges.map(function(bridge) {
      const pts = Array.isArray(bridge.points) ? bridge.points : [];
      return pts.length ? [Number(pts[0].x) || 0, Number(pts[0].y) || 0] : null;
    }).filter(Boolean);
    if (starts.length) {
      let sx = 0, sy = 0;
      starts.forEach(function(pt) { sx += pt[0]; sy += pt[1]; });
      return [sx / starts.length, sy / starts.length];
    }
    return [0, 0];
  }
  function getPBBStandAngle(pbb) {
    if (pbb && pbb.angleDeg != null) return normalizeAngleDeg(pbb.angleDeg) * Math.PI / 180;
    const anchor = getPbbAnchorPx(pbb);
    const center = getStandConnectionPx(pbb);
    return Math.atan2(center[1] - anchor[1], center[0] - anchor[0]);
  }
  function getPBBStandCorners(pbb) {
    const center = getStandConnectionPx(pbb);
    const cx = center[0], cy = center[1];
    const cat = pbb.category || 'C';
    const dep = getStandDepthMeters(cat);
    const halfD = dep / 2;
    const halfW = getStandWidthMeters(cat) / 2;
    const angle = getPBBStandAngle(pbb);
    const shiftX = standStopbarCenterShiftLocalX(dep, cat);
    const cos = Math.cos(angle), sin = Math.sin(angle);
    return [
      [cx + ((-halfD + shiftX))*cos - (-halfW)*sin, cy + ((-halfD + shiftX))*sin + (-halfW)*cos],
      [cx + (( halfD + shiftX))*cos - (-halfW)*sin, cy + (( halfD + shiftX))*sin + (-halfW)*cos],
      [cx + (( halfD + shiftX))*cos - ( halfW)*sin, cy + (( halfD + shiftX))*sin + ( halfW)*cos],
      [cx + ((-halfD + shiftX))*cos - ( halfW)*sin, cy + ((-halfD + shiftX))*sin + ( halfW)*cos]
    ];
  }
  function pointInPolygonXY(p, verts) {
    let inside = false;
    const n = verts.length;
    for (let i = 0, j = n - 1; i < n; j = i++) {
      const vi = verts[i], vj = verts[j];
      if (((vi[1] > p[1]) !== (vj[1] > p[1])) && (p[0] < (vj[0]-vi[0])*(p[1]-vi[1])/(vj[1]-vi[1])+vi[0])) inside = !inside;
    }
    return inside;
  }
  function segIntersect(a1, a2, b1, b2) {
    const [ax1,ay1]=a1,[ax2,ay2]=a2,[bx1,by1]=b1,[bx2,by2]=b2;
    const dax = ax2-ax1, day = ay2-ay1, dbx = bx2-bx1, dby = by2-by1;
    const den = dax*dby - day*dbx;
    if (Math.abs(den) < 1e-10) return false;
    const t = ((bx1-ax1)*dby - (by1-ay1)*dbx) / den;
    const s = ((bx1-ax1)*day - (by1-ay1)*dax) / den;
    return t >= 0 && t <= 1 && s >= 0 && s <= 1;
  }
  function rotatedRectsOverlap(cornersA, cornersB) {
    for (let i = 0; i < 4; i++) if (pointInPolygonXY(cornersA[i], cornersB)) return true;
    for (let i = 0; i < 4; i++) if (pointInPolygonXY(cornersB[i], cornersA)) return true;
    for (let i = 0; i < 4; i++) {
      const a1 = cornersA[i], a2 = cornersA[(i+1)%4];
      for (let j = 0; j < 4; j++) {
        if (segIntersect(a1, a2, cornersB[j], cornersB[(j+1)%4])) return true;
      }
    }
    return false;
  }
  function distPointToSegment(px, py, ax, ay, bx, by) {
    const vx = bx - ax, vy = by - ay;
    const wx = px - ax, wy = py - ay;
    const c1 = vx * wx + vy * wy;
    if (c1 <= 0) return Math.hypot(px - ax, py - ay);
    const c2 = vx * vx + vy * vy;
    if (c2 <= c1) return Math.hypot(px - bx, py - by);
    const t = c1 / c2;
    const projx = ax + t * vx, projy = ay + t * vy;
    return Math.hypot(px - projx, py - projy);
  }
  function minDistanceConvexQuads(quadA, quadB) {
    if (rotatedRectsOverlap(quadA, quadB)) return 0;
    let minD = Infinity;
    for (let i = 0; i < 4; i++) {
      const p = quadA[i];
      for (let j = 0; j < 4; j++) {
        const q1 = quadB[j], q2 = quadB[(j + 1) % 4];
        minD = Math.min(minD, distPointToSegment(p[0], p[1], q1[0], q1[1], q2[0], q2[1]));
      }
    }
    for (let i = 0; i < 4; i++) {
      const p = quadB[i];
      for (let j = 0; j < 4; j++) {
        const q1 = quadA[j], q2 = quadA[(j + 1) % 4];
        minD = Math.min(minD, distPointToSegment(p[0], p[1], q1[0], q1[1], q2[0], q2[1]));
      }
    }
    return minD;
  }
  function standFootprintsTooClose(cornersA, catA, cornersB, catB) {
    const need = getStandSpacingMeters(catA, catB);
    if (need <= 0) return rotatedRectsOverlap(cornersA, cornersB);
    return minDistanceConvexQuads(cornersA, cornersB) < need;
  }
  function buildStandSafetyPolygonLocalPoints(depM, widM, category) {
    const r = standConfigRowForIcaoCat(category);
    if (!r || !isFinite(depM) || !isFinite(widM) || depM <= 0 || widM <= 0) return null;
    const nw = Number(r.nose_width), nc = Number(r.nose_clear);
    if (!isFinite(nw) || nw <= 0 || !isFinite(nc) || nc <= 0) return null;
    const halfD = depM / 2, halfW = widM / 2;
    const noseHalf = nw / 2;
    const eps = 0.08;
    if (noseHalf >= halfW - eps) return null;
    const xNose = -halfD;
    const xStop = -halfD + nc;
    if (xStop <= xNose + eps || xStop >= halfD - eps) return null;
    const latRun = halfW - noseHalf;
    if (latRun <= eps) return null;
    const xBendEnd = xStop + latRun;
    if (xBendEnd > halfD + eps) return null;
    const pts = [];
    pts.push([xNose, -noseHalf]);
    pts.push([xNose, noseHalf]);
    pts.push([xStop, noseHalf]);
    pts.push([Math.min(xBendEnd, halfD), halfW]);
    if (xBendEnd < halfD - eps) {
      pts.push([halfD, halfW]);
      pts.push([halfD, -halfW]);
      pts.push([xBendEnd, -halfW]);
    } else {
      pts.push([halfD, -halfW]);
    }
    pts.push([xStop, -noseHalf]);
    return pts;
  }
  function standOuterContourWorldPolygonForSpec(cx, cy, angleRad, depM, widM, category) {
    const polyLocal = buildStandSafetyPolygonLocalPoints(depM, widM, category);
    if (polyLocal && polyLocal.length >= 3) {
      return polyLocal.map(function(p) { return standFootprintLocalToWorld(cx, cy, angleRad, p[0], p[1]); });
    }
    return [
      standFootprintLocalToWorld(cx, cy, angleRad, -depM / 2, -widM / 2),
      standFootprintLocalToWorld(cx, cy, angleRad, depM / 2, -widM / 2),
      standFootprintLocalToWorld(cx, cy, angleRad, depM / 2, widM / 2),
      standFootprintLocalToWorld(cx, cy, angleRad, -depM / 2, widM / 2),
    ];
  }
  function standGapSegmentsWorldForSpec(cx, cy, angleRad, depM, widM, category) {
    const r = standConfigRowForIcaoCat(category);
    if (!r || !isFinite(depM) || !isFinite(widM) || depM <= 0 || widM <= 0) return [];
    const g = Number(r.gap), ws = Number(r.wingspan);
    const halfD = depM / 2, halfW = widM / 2;
    const eps = 0.12;
    if (!(isFinite(g) && g > eps && isFinite(ws) && ws > 0)) return [];
    const yLim = halfW - g;
    if (!(yLim > eps && yLim < halfW - eps)) return [];
    const a0 = standFootprintLocalToWorld(cx, cy, angleRad, -halfD, yLim);
    const a1 = standFootprintLocalToWorld(cx, cy, angleRad, halfD, yLim);
    const b0 = standFootprintLocalToWorld(cx, cy, angleRad, -halfD, -yLim);
    const b1 = standFootprintLocalToWorld(cx, cy, angleRad, halfD, -yLim);
    return [[a0, a1], [b0, b1]];
  }
  function standGapSegmentsIntersectOuterPolygon(gapSegs, polygon) {
    if (!Array.isArray(gapSegs) || !gapSegs.length || !Array.isArray(polygon) || polygon.length < 2) return false;
    for (let i = 0; i < gapSegs.length; i++) {
      const seg = gapSegs[i];
      if (!seg || seg.length < 2) continue;
      const g0 = seg[0], g1 = seg[1];
      for (let j = 0; j < polygon.length; j++) {
        const p0 = polygon[j], p1 = polygon[(j + 1) % polygon.length];
        if (segIntersect(g0, g1, p0, p1)) return true;
      }
    }
    return false;
  }
  function standGapLineHitsExistingOuterContours(candidateCenter, candidateAngleRad, candidateCategory) {
    const depC = getStandDepthMeters(candidateCategory || 'C');
    const widC = getStandWidthMeters(candidateCategory || 'C');
    const gapSegs = standGapSegmentsWorldForSpec(candidateCenter[0], candidateCenter[1], candidateAngleRad, depC, widC, candidateCategory || 'C');
    if (!gapSegs.length) return false;
    function hitWithPolygon(poly) { return standGapSegmentsIntersectOuterPolygon(gapSegs, poly); }
    for (let i = 0; i < state.remoteStands.length; i++) {
      const o = state.remoteStands[i];
      const oc = getRemoteStandCenterPx(o);
      const oa = getRemoteStandAngleRad(o);
      const od = getStandDepthMeters(o.category || 'C');
      const ow = getStandWidthMeters(o.category || 'C');
      if (hitWithPolygon(standOuterContourWorldPolygonForSpec(oc[0], oc[1], oa, od, ow, o.category || 'C'))) return true;
    }
    const temps = state.tempStands || [];
    for (let i = 0; i < temps.length; i++) {
      const o = temps[i];
      const oc = getRemoteStandCenterPx(o);
      const oa = getRemoteStandAngleRad(o);
      const od = getStandDepthMeters(o.category || 'C');
      const ow = getStandWidthMeters(o.category || 'C');
      if (hitWithPolygon(standOuterContourWorldPolygonForSpec(oc[0], oc[1], oa, od, ow, o.category || 'C'))) return true;
    }
    for (let i = 0; i < state.pbbStands.length; i++) {
      const o = state.pbbStands[i];
      const oc = getStandConnectionPx(o);
      const oa = getPBBStandAngle(o);
      const od = getStandDepthMeters(o.category || 'C');
      const ow = getStandWidthMeters(o.category || 'C');
      if (hitWithPolygon(standOuterContourWorldPolygonForSpec(oc[0], oc[1], oa, od, ow, o.category || 'C'))) return true;
    }
    return false;
  }
  function pbbStandOverlapsTerminal(pbb) {
    const corners = getPBBStandCorners(pbb);
    for (let t = 0; t < state.terminals.length; t++) {
      const term = state.terminals[t];
      if (!term.closed || term.vertices.length < 3) continue;
      const termPix = term.vertices.map(v => cellToPixel(v.col, v.row));
      for (let k = 0; k < 4; k++) {
        if (pointInPolygonXY(corners[k], termPix)) return true;
      }
      for (let k = 0; k < termPix.length; k++) {
        if (pointInPolygonXY(termPix[k], corners)) return true;
      }
    }
    return false;
  }
  function pbbStandOverlapsExisting(pbb, excludeId) {
    if (pbbStandOverlapsTerminal(pbb)) return true;
    const cat = pbb.category || 'C';
    const center = getStandConnectionPx(pbb);
    const angle = getPBBStandAngle(pbb);
    if (standGapLineHitsExistingOuterContours(center, angle, cat)) return true;
    return false;
  }
  function pbbStandOuterContoursOverlapExisting(pbb, excludeId) {
    const corners = getPBBStandCorners(pbb);
    for (let i = 0; i < state.pbbStands.length; i++) {
      const other = state.pbbStands[i];
      if (!other) continue;
      if (excludeId && other.id === excludeId) continue;
      if (rotatedRectsOverlap(corners, getPBBStandCorners(other))) return true;
    }
    for (let i = 0; i < state.remoteStands.length; i++) {
      const st = state.remoteStands[i];
      if (!st) continue;
      if (rotatedRectsOverlap(corners, getRemoteStandCorners(st))) return true;
    }
    const temps = state.tempStands || [];
    for (let i = 0; i < temps.length; i++) {
      const st = temps[i];
      if (!st) continue;
      if (rotatedRectsOverlap(corners, getRemoteStandCorners(st))) return true;
    }
    return false;
  }
  function tryPlacePbbAt(wx, wy) {
    let bestEdge = null, bestD2 = Infinity;
    state.terminals.forEach(t => {
      if (!t.closed || t.vertices.length < 2) return;
      let cx = 0, cy = 0;
      t.vertices.forEach(v => { const [px, py] = cellToPixel(v.col, v.row); cx += px; cy += py; });
      cx /= t.vertices.length || 1; cy /= t.vertices.length || 1;
      for (let i = 0; i < t.vertices.length; i++) {
        const v1 = t.vertices[i], v2 = t.vertices[(i + 1) % t.vertices.length];
        const p1 = cellToPixel(v1.col, v1.row), p2 = cellToPixel(v2.col, v2.row);
        const near = closestPointOnSegment(p1, p2, [wx, wy]);
        if (near) {
          const d2 = dist2(near, [wx, wy]);
          if (d2 < bestD2) { bestD2 = d2; bestEdge = { near, p1, p2, col: v1.col, row: v1.row, cx, cy }; }
        }
      }
    });
    const maxD2 = (CELL_SIZE * TRY_PBB_MAX_EDGE_CF) ** 2;
    if (!bestEdge || bestD2 >= maxD2) return false;
    const [ex, ey] = bestEdge.near, [x1, y1] = bestEdge.p1, [x2, y2] = bestEdge.p2;
    let nx = -(y2 - y1), ny = x2 - x1;
    const len = Math.hypot(nx, ny) || 1; nx /= len; ny /= len;
    const inX = bestEdge.cx - ex, inY = bestEdge.cy - ey;
    if (nx * inX + ny * inY > 0) { nx *= -1; ny *= -1; }
    const uPbb = readUnifiedNewStandConstraintFromPanel('standIcaoCategories', 'standAircraftAccess', ['A', 'B', 'C']);
    const categoryMode = uPbb.categoryMode;
    const category = uPbb.category;
    const allowedIcaoCategories = uPbb.allowedIcaoCategories;
    const panelAllowedTypes = uPbb.allowedAircraftTypes;
    const minLen = getStandDepthMeters(category) / 2 + 3;
    const lenMeters = Number(document.getElementById('pbbLength').value || 15);
    const armLen = Math.max(isFinite(lenMeters) && lenMeters > 0 ? lenMeters : 15, minLen);
    const standAngleDeg = normalizeAngleDeg(Math.atan2(ny, nx) * 180 / Math.PI);
    const bwEl = document.getElementById('pbbBoardingWidth');
    const bhEl = document.getElementById('pbbBoardingHeight');
    const boardingW = Math.max(0.5, Number(bwEl && bwEl.value) || 5);
    const boardingH = Math.max(0.5, Number(bhEl && bhEl.value) || 15);
    const wallX = ex, wallY = ey;
    const bxOut = wallX + nx * boardingH, byOut = wallY + ny * boardingH;
    const cfgRow = standConfigRowForIcaoCat(category);
    const noseClear = cfgRow ? Number(cfgRow.nose_clear) : NaN;
    const offM = (Number.isFinite(noseClear) && noseClear > 0)
      ? noseClear
      : PBB_STAND_CENTER_OFFSET_FROM_TERMINAL_WALL_M;
    const newPbb = {
      x1: wallX, y1: wallY, x2: bxOut, y2: byOut, category,
      angleDeg: standAngleDeg,
      apronSiteX: wallX + nx * offM,
      apronSiteY: wallY + ny * offM,
      terminalContactSetbackM: offM,
      boardingWidthM: boardingW,
      boardingHeightM: boardingH
    };
    if (pbbStandOverlapsExisting(newPbb)) return false;
    const pbbNameCandidate = document.getElementById('standName').value.trim() || getDefaultPbbStandName();
    if (findDuplicateLayoutName('pbb', null, pbbNameCandidate)) {
      alertDuplicateLayoutName();
      return false;
    }
    pushUndo();
    state.pbbStands.push(normalizePbbStandObject({
      id: id(),
      name: pbbNameCandidate,
      x1: wallX, y1: wallY, x2: bxOut, y2: byOut,
      category: newPbb.category,
      terminalContactSetbackM: offM,
      categoryMode: categoryMode,
      allowedIcaoCategories: allowedIcaoCategories,
      allowedAircraftTypes: panelAllowedTypes,
      pbbCount: Math.max(1, Math.min(8, parseInt(document.getElementById('pbbBridgeCount') ? document.getElementById('pbbBridgeCount').value : (_pbbTier.defaultBridgeCount || 1), 10) || 1)),
      angleDeg: standAngleDeg,
      apronSiteX: newPbb.apronSiteX,
      apronSiteY: newPbb.apronSiteY,
      boardingWidthM: boardingW,
      boardingHeightM: boardingH,
      pbbArmLenM: armLen,
      edgeCol: bestEdge.col,
      edgeRow: bestEdge.row
    }));
    return true;
  }
  function tryPlaceRemoteAt(wx, wy) {
    if (!isFinite(wx) || !isFinite(wy)) return false;
    const maxX = GRID_COLS * CELL_SIZE, maxY = GRID_ROWS * CELL_SIZE;
    if (wx < 0 || wy < 0 || wx > maxX || wy > maxY) return false;
    const uRm = readUnifiedNewStandConstraintFromPanel('remoteIcaoCategories', 'remoteAircraftAccess', ['A', 'B', 'C']);
    const categoryMode = uRm.categoryMode;
    const category = uRm.category;
    const allowedIcaoCategoriesR = uRm.allowedIcaoCategories;
    const panelAllowedTypesR = uRm.allowedAircraftTypes;
    const angleDeg = 0;
    const candidate = { x: Number(wx), y: Number(wy), category, angleDeg };
    const candCorners = getRemoteStandCorners(candidate);
    for (let i = 0; i < state.remoteStands.length; i++) {
      const o = state.remoteStands[i];
      if (standFootprintsTooClose(candCorners, category, getRemoteStandCorners(o), o.category || 'C')) return false;
    }
    for (let i = 0; i < state.pbbStands.length; i++) {
      const o = state.pbbStands[i];
      if (standFootprintsTooClose(candCorners, category, getPBBStandCorners(o), o.category || 'C')) return false;
    }
    for (let i = 0; i < (state.tempStands || []).length; i++) {
      const o = state.tempStands[i];
      if (standFootprintsTooClose(candCorners, category, getRemoteStandCorners(o), o.category || 'C')) return false;
    }
    if (standGapLineHitsExistingOuterContours([Number(wx), Number(wy)], angleDeg * Math.PI / 180, category)) return false;
    const baseName = (document.getElementById('remoteName') && document.getElementById('remoteName').value.trim()) || getDefaultRemoteStandName();
    if (findDuplicateLayoutName('remote', null, baseName)) {
      alertDuplicateLayoutName();
      return false;
    }
    pushUndo();
    state.remoteStands.push(normalizeRemoteStandObject({
      id: id(),
      x: Number(wx),
      y: Number(wy),
      category,
      name: baseName,
      angleDeg,
      categoryMode: categoryMode,
      allowedIcaoCategories: allowedIcaoCategoriesR,
      allowedAircraftTypes: panelAllowedTypesR,
      allowedTerminals: Array.from((document.getElementById('remoteTerminalAccess') || document).querySelectorAll('.remote-term-check')).filter(function(ch) { return ch.checked; }).map(function(ch) { return String(ch.getAttribute('data-item-id') || '').trim(); }).filter(Boolean)
    }));
    return true;
  }
  function tryPlaceTempStandAt(wx, wy) {
    const snap = snapTempStandOnTaxiwayCenterlines(wx, wy);
    if (!snap) return false;
    const sx = snap.x, sy = snap.y;
    const uTs = readUnifiedNewStandConstraintFromPanel('tempStandIcaoCategories', 'tempStandAircraftAccess', ['A', 'B', 'C']);
    const categoryMode = uTs.categoryMode;
    const category = uTs.category;
    const allowedIcaoCategoriesT = uTs.allowedIcaoCategories;
    const panelAllowedTypesT = uTs.allowedAircraftTypes;
    const angleDeg = 0;
    const candidate = { x: Number(sx), y: Number(sy), category, angleDeg };
    const candCorners = getRemoteStandCorners(candidate);
    for (let i = 0; i < (state.tempStands || []).length; i++) {
      const o = state.tempStands[i];
      if (standFootprintsTooClose(candCorners, category, getRemoteStandCorners(o), o.category || 'C')) return false;
    }
    for (let i = 0; i < state.remoteStands.length; i++) {
      const o = state.remoteStands[i];
      if (standFootprintsTooClose(candCorners, category, getRemoteStandCorners(o), o.category || 'C')) return false;
