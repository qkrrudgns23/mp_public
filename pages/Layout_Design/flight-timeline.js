      const sub = document.createElement('div');
      sub.className = 'layout-toast-sub';
      sub.textContent = String(subText);
      el.appendChild(sub);
    }
    stack.appendChild(el);
    requestAnimationFrame(function() { el.classList.add('is-visible'); });
    const lifeMs = kind === 'error' ? 4200 : 2600;
    setTimeout(function() {
      el.classList.remove('is-visible');
      setTimeout(function() { if (el.parentNode === stack) stack.removeChild(el); }, 220);
    }, lifeMs);
  }
  /**
   * Render apron site anchor as a 5m × 1m rectangle. Long side (5m) is aligned
   * with the stand's stopbar direction (local Y = perpendicular to the stand
   * depth axis), so it matches the dashed stopbar line drawn near the nose.
   * ``standAngleRad`` is the stand's depth-axis angle (same as ctx.rotate used
   * when drawing the stand footprint). When omitted, the 5m side falls back to
   * world horizontal.
   */
  function drawApronSiteMarker(ctx, cx, cy, fillStyle, strokeStyle, selected, standAngleRad) {
    const w = APRON_SITE_MARKER_WIDTH_M;
    const h = APRON_SITE_MARKER_HEIGHT_M;
    const angle = (typeof standAngleRad === 'number' && isFinite(standAngleRad))
      ? standAngleRad + Math.PI / 2
      : 0;
    ctx.save();
    ctx.translate(cx, cy);
    if (angle) ctx.rotate(angle);
    ctx.beginPath();
    ctx.rect(-w / 2, -h / 2, w, h);
    if (fillStyle) {
      ctx.fillStyle = fillStyle;
      ctx.fill();
    }
    if (strokeStyle) {
      const baseLw = Math.max(0.18, 0.24 / Math.max(state.scale, 0.1));
      ctx.lineWidth = selected ? baseLw * 1.5 : baseLw;
      ctx.strokeStyle = strokeStyle;
      ctx.stroke();
    }
    ctx.restore();
  }
  /** Apron–taxiway UI attach point: PBB = aircraft marker; remote/temp = same local xBendEnd offset as Contact (getStandAircraftMarkerWorldPxFor*). */
  function getStandApronTaxiwayAttachWorldPx(stand) {
    if (!stand) return [0, 0];
    const isPbb = (state.pbbStands || []).some(function(s) { return s && s.id === stand.id; });
    if (isPbb) return getStandConnectionPx(stand);
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
    const offM = PBB_NEW_CONTACT_STAND_SITE_OFFSET_M;
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
    }
    for (let i = 0; i < state.pbbStands.length; i++) {
      const o = state.pbbStands[i];
      if (standFootprintsTooClose(candCorners, category, getPBBStandCorners(o), o.category || 'C')) return false;
    }
    if (standGapLineHitsExistingOuterContours([Number(sx), Number(sy)], angleDeg * Math.PI / 180, category)) return false;
    const baseName = (document.getElementById('tempStandName') && document.getElementById('tempStandName').value.trim()) || getDefaultTempStandName();
    if (findDuplicateLayoutName('tempStand', null, baseName)) {
      alertDuplicateLayoutName();
      return false;
    }
    pushUndo();
    state.tempStands.push(normalizeTempStandObject({
      id: id(),
      x: Number(sx),
      y: Number(sy),
      junctionX: Number(sx),
      junctionY: Number(sy),
      category,
      name: baseName,
      angleDeg,
      categoryMode: categoryMode,
      allowedIcaoCategories: allowedIcaoCategoriesT,
      allowedAircraftTypes: panelAllowedTypesT,
      allowedTerminals: Array.from((document.getElementById('tempStandTerminalAccess') || document).querySelectorAll('.remote-term-check')).filter(function(ch) { return ch.checked; }).map(function(ch) { return String(ch.getAttribute('data-item-id') || '').trim(); }).filter(Boolean)
    }));
    return true;
  }
  function taxiwayOverlapsAnyTerminal(tw) {
    if (!tw || !tw.vertices || tw.vertices.length < 2) return false;
    const vertsPix = tw.vertices.map(v => cellToPixel(v.col, v.row));
    for (let t = 0; t < state.terminals.length; t++) {
      const term = state.terminals[t];
      if (!term.closed || term.vertices.length < 3) continue;
      const termPix = term.vertices.map(v => cellToPixel(v.col, v.row));
      for (let i = 0; i < vertsPix.length; i++) {
        if (pointInPolygonXY(vertsPix[i], termPix)) return true;
      }
      for (let i = 0; i < vertsPix.length - 1; i++) {
        const a1 = vertsPix[i], a2 = vertsPix[i+1];
        for (let j = 0; j < termPix.length; j++) {
          const b1 = termPix[j], b2 = termPix[(j+1) % termPix.length];
          if (segIntersect(a1, a2, b1, b2)) return true;
        }
      }
    }
    return false;


  }
  function terminalOverlapsAnyTaxiway(term) {
    if (!term || !term.vertices || term.vertices.length < 3) return false;
    const termPix = term.vertices.map(v => cellToPixel(v.col, v.row));
    if (!state.taxiways || !state.taxiways.length) return false;
    for (let i = 0; i < state.taxiways.length; i++) {
      const tw = state.taxiways[i];
      if (!tw.vertices || tw.vertices.length < 2) continue;
      const vertsPix = tw.vertices.map(v => cellToPixel(v.col, v.row));
      for (let k = 0; k < vertsPix.length; k++) {
        if (pointInPolygonXY(vertsPix[k], termPix)) return true;
      }
      for (let a = 0; a < vertsPix.length - 1; a++) {
        const a1 = vertsPix[a], a2 = vertsPix[a+1];
        for (let b = 0; b < termPix.length; b++) {
          const b1 = termPix[b], b2 = termPix[(b+1) % termPix.length];
          if (segIntersect(a1, a2, b1, b2)) return true;
        }
      }
    }
    return false;
  }
  function makeUniqueNamedCopy(list, _prop) {
    return (list || []).map(function(obj) {
      return Object.assign({}, obj);
    });
  }

  function _persistCellSizePx() {
    return (typeof CELL_SIZE === 'number' && CELL_SIZE > 0) ? CELL_SIZE : 20;
  }
  function persistVerticesCellsToXY(vertices) {
    const cs = _persistCellSizePx();
    if (!Array.isArray(vertices)) return [];
    return vertices.map(function(v) {
      if (!v || typeof v !== 'object') return { x: 0, y: 0 };
      const c = Number(v.col), r = Number(v.row);
      return { x: (isFinite(c) ? c : 0) * cs, y: (isFinite(r) ? r : 0) * cs };
    });
  }
  function persistPointCellToXY(pt) {
    if (!pt || typeof pt !== 'object') return null;
    const xRaw = Number(pt.x), yRaw = Number(pt.y);
    if (isFinite(xRaw) && isFinite(yRaw)) return { x: xRaw, y: yRaw };
    const cs = _persistCellSizePx();
    const c = Number(pt.col), r = Number(pt.row);
    return { x: (isFinite(c) ? c : 0) * cs, y: (isFinite(r) ? r : 0) * cs };
  }

  function _polylineLengthPxForLineup(pts) {
    if (!pts || pts.length < 2) return 0;
    let s = 0;
    for (let i = 0; i < pts.length - 1; i++) {
      const p1 = pts[i], p2 = pts[i + 1];
      s += Math.hypot(p2[0] - p1[0], p2[1] - p1[1]);
    }
    return s;
  }
  function _pointOnPolylineAtDistPxForLineup(pts, distPx) {
    if (!pts || pts.length < 2) return null;
    const total = _polylineLengthPxForLineup(pts);
    const d = Math.max(0, Math.min(typeof distPx === 'number' ? distPx : 0, total));
    let acc = 0;
    for (let i = 0; i < pts.length - 1; i++) {
      const p1 = pts[i], p2 = pts[i + 1];
      const segLen = Math.hypot(p2[0] - p1[0], p2[1] - p1[1]);
      if (!(segLen > 1e-6)) continue;
      if (acc + segLen >= d - 1e-6) {
        const t = Math.max(0, Math.min(1, (d - acc) / segLen));
        return [p1[0] + (p2[0] - p1[0]) * t, p1[1] + (p2[1] - p1[1]) * t];
      }
      acc += segLen;
    }
    const last = pts[pts.length - 1];
    return [last[0], last[1]];
  }
  /** Ordered runway polyline in layout px (matches getRunwayPath / departure graphPath). */
  function _persistRunwayPolylinePtsPx(tw) {
    if (!tw || tw.pathType !== 'runway' || !tw.vertices || tw.vertices.length < 2) return null;
    return tw.vertices.map(function(v) { return cellToPixel(v.col, v.row); });
  }

  function serializeTaxiwayWithEndpoints(tw) {
    const copy = Object.assign({}, tw);
    if (Array.isArray(tw.vertices)) {
      copy.vertices = persistVerticesCellsToXY(tw.vertices.slice());
    }
    delete copy.start_point;
    delete copy.end_point;
    if (typeof tw.avgMoveVelocity === 'number' && isFinite(tw.avgMoveVelocity) && tw.avgMoveVelocity > 0) {
      copy.avgMoveVelocity = tw.avgMoveVelocity;
    }
    if (tw.pathType === 'runway' && typeof tw.minArrVelocity === 'number' && isFinite(tw.minArrVelocity) && tw.minArrVelocity > 0) {
      copy.minArrVelocity = Math.max(1, Math.min(150, tw.minArrVelocity));
    }
    if (tw.pathType === 'runway') {
      const lCw = getRunwayLineupDistMByDirection(tw, 'clockwise');
      const lCcw = getRunwayLineupDistMByDirection(tw, 'counter_clockwise');
      copy.lineupDistM_CW = lCw;
      copy.lineupDistM_CCW = lCcw;
      copy.lineupDistM = getEffectiveRunwayLineupDistM(tw);
      if (typeof tw.startDisplacedThresholdM === 'number' && isFinite(tw.startDisplacedThresholdM) && tw.startDisplacedThresholdM >= 0) copy.startDisplacedThresholdM = tw.startDisplacedThresholdM;
      else delete copy.startDisplacedThresholdM;
      if (typeof tw.startBlastPadM === 'number' && isFinite(tw.startBlastPadM) && tw.startBlastPadM >= 0) copy.startBlastPadM = tw.startBlastPadM;
      else delete copy.startBlastPadM;
      if (typeof tw.endDisplacedThresholdM === 'number' && isFinite(tw.endDisplacedThresholdM) && tw.endDisplacedThresholdM >= 0) copy.endDisplacedThresholdM = tw.endDisplacedThresholdM;
      else delete copy.endDisplacedThresholdM;
      if (typeof tw.endBlastPadM === 'number' && isFinite(tw.endBlastPadM) && tw.endBlastPadM >= 0) copy.endBlastPadM = tw.endBlastPadM;
      else delete copy.endBlastPadM;
      const rwPts = _persistRunwayPolylinePtsPx(tw);
      if (rwPts && rwPts.length >= 2) {
        const lenPx = _polylineLengthPxForLineup(rwPts);
        const dPx = getEffectiveRunwayLineupDistFromStartM(tw, lenPx);
        const lp = _pointOnPolylineAtDistPxForLineup(rwPts, dPx);
        if (lp) copy.lineup_point = { x: lp[0], y: lp[1] };
        else delete copy.lineup_point;
      } else {
        delete copy.lineup_point;
      }
      delete copy.dep_point;
      delete copy.depPointPos;
    }
    if (tw.pathType === 'runway' && tw.rwySepConfig) copy.rwySepConfig = tw.rwySepConfig;
    else delete copy.rwySepConfig;
    return copy;
  }
  function partitionTaxiwaysForPersist(list) {
    const runwayPaths = [];
    const runwayTaxiways = [];
    const taxiways = [];
    (list || []).forEach(function(tw) {
      const ser = serializeTaxiwayWithEndpoints(tw);
      const pt = tw.pathType || 'taxiway';
      if (pt === 'runway') runwayPaths.push(ser);
      else if (pt === 'runway_exit') runwayTaxiways.push(ser);
      else {
        if (pt === 'general_queue_taxiway') ser.pathType = pt;
        else delete ser.pathType;
        taxiways.push(ser);
      }
    });
    return { runwayPaths: runwayPaths, runwayTaxiways: runwayTaxiways, taxiways: taxiways };
  }
  function serializeCurrentLayout() {
    function pathJunctionsToNetworkJunctions(pts) {
      const out = [];
      (pts || []).forEach(function(p) {
        if (!p) return;
        if (Array.isArray(p) && p.length >= 2) {
          out.push({ x: p[0], y: p[1] });
        } else if (typeof p.x === 'number' && typeof p.y === 'number') {
          out.push({ x: p.x, y: p.y });
        }
      });
      return out;
    }
    let networkJunctions = pathJunctionsToNetworkJunctions(state.pathGraphJunctions);
    if (!networkJunctions.length && typeof buildPathGraph === 'function') {
      try {
        let gj = null;
        const sig = computeTaxiwaysGraphSig();
        if (state.pathGraphCacheValid && state.pathGraphCache && state.pathGraphCacheSig === sig) {
          gj = state.pathGraphCache;
        } else if (!PATH_GRAPH_SYNC_ONLY_ON_EXPLICIT_ACTION) {
          gj = buildPathGraph(null);
        }
        if (gj) {
          const cj = (gj && (gj.connectedJunctions || gj.junctions)) || [];
          networkJunctions = pathJunctionsToNetworkJunctions(cj);
        }
      } catch (e) { /* ignore */ }
    }
    let edgeExport = [];
    if (typeof rebuildDerivedGraphEdges === 'function') {
      try {
        rebuildDerivedGraphEdges();
        edgeExport = (state.derivedGraphEdges || []).map(function(ed) {
          return { id: ed.id, label: ed.label, name: ed.name, fromIdx: ed.fromIdx, toIdx: ed.toIdx };
        });
      } catch (e2) { edgeExport = []; }
    }
    return {
      grid: {
        cols: GRID_COLS,
        rows: GRID_ROWS,
        cellSize: CELL_SIZE,
        showGrid: !!state.showGrid,
        showImage: !!state.showImage,
        showRoadWidth: !!state.showRoadWidth,
        showLayoutMarkers: !!state.showLayoutMarkers,
        layers: Object.assign({}, state.layers),
        layerMono: state.layerMono ? Object.assign({}, state.layerMono) : Object.assign({}, DEFAULT_LAYER_MONO),
        layoutImageOverlay: state.layoutImageOverlay ? Object.assign({}, state.layoutImageOverlay) : null
      },
      networkJunctions: networkJunctions,
      Edge: edgeExport,
      terminals: makeUniqueNamedCopy(state.terminals, 'name').map(function(t) {
        const o = Object.assign({}, t);
        if (Array.isArray(o.vertices)) o.vertices = persistVerticesCellsToXY(o.vertices);
        return o;
      }),
      pbbStands: makeUniqueNamedCopy(state.pbbStands, 'name'),
      remoteStands: state.remoteStands.slice(),
      tempStands: (state.tempStands || []).slice(),
      holdingPoints: (state.holdingPoints || []).slice(),
      ...(function() {
        const p = partitionTaxiwaysForPersist(state.taxiways);
        return { runwayPaths: p.runwayPaths, runwayTaxiways: p.runwayTaxiways, taxiways: p.taxiways };
      })(),
      apronLinks: (state.apronLinks || []).map(function(lk) {
        const o = Object.assign({}, lk);
        if (Array.isArray(o.midVertices)) o.midVertices = persistVerticesCellsToXY(o.midVertices);
        return o;
      }),
      directionModes: state.directionModes.slice(),
      flights: state.flights.map(function(f) {
        const copy = {};
        const simFlightKeys = [
          'id',
          'reg',
          'airlineCode',
          'flightNumber',
          'aircraftType',
          'code',
          'timeMin',
          'sibtDate',
          'dwellMin',
          'minDwellMin',
          'noWayArr',
          'noWayDep',
          'arrRetFailed',
          'serviceDate',
          'sldtMin',
          'sibtMin',
          'sobtMin',
          'stotMin',
          'arrApronId',
          'depApronId',
          'eibtMinList',
          'eobtMinList',
          'ePushFinishedMinList',
          'arrRunwayDirUsed',
          'depRunwayDirUsed',
          'arrTdDistM',
          'arrVTdMs',
          'arrDecelMs2',
          'arrDep',
          'intDom',
          'arrRotSec',
          'proSimVttArrSec',
          'proSimDttArrSec',
          'proSimPushbackSec',
          'proSimDttDepSec',
          'proSimVttDepSec',
          'proSimDepLineupSec',
          'arrRunwayIdUsed',
          'arrRetDistM',
          'arrVRetInMs',
          'arrVRetOutMs',
        ];
        simFlightKeys.forEach(function(k) {
          if (Object.prototype.hasOwnProperty.call(f, k) && f[k] !== undefined) {
            copy[k] = f[k];
          }
        });
        const apronStaySegments = (typeof serializableApronStaySegmentsForFlight === 'function')
          ? serializableApronStaySegmentsForFlight(f)
          : [];
        if (apronStaySegments.length) {
          copy.apronStaySegments = apronStaySegments.map(function(seg) {
            const out = { sibtMin: seg.sibtMin, sobtMin: seg.sobtMin };
            if (seg.standId != null) out.standId = seg.standId;
            return out;
          });
          copy.arrApronId = copy.apronStaySegments[0].standId || null;
          copy.depApronId = copy.apronStaySegments[copy.apronStaySegments.length - 1].standId || null;
          copy.standId = copy.depApronId || null;
        }
        if (Array.isArray(f.edge_list) && f.edge_list.length) {
          copy.edge_list = f.edge_list.slice();
        }
        const t = f.token || {};
        const arrRwyId = f.arrRunwayId || t.arrRunwayId || t.runwayId || null;
        const apronId = (copy.depApronId != null ? copy.depApronId : (f.standId != null ? f.standId : (t.apronId != null ? t.apronId : null)));
        const termId = f.terminalId || t.terminalId || null;
        const depRwyId = f.depRunwayId || t.depRunwayId || null;
        const exitTwId = (f.sampledArrRet != null && f.sampledArrRet !== '') ? f.sampledArrRet : (t.ExitTaxiwayId != null ? t.ExitTaxiwayId : null);
        copy.token = {
          arrRunwayId: arrRwyId,
          ExitTaxiwayId: exitTwId || null,
          apronId: apronId || null,
          terminalId: termId || null,
          depRunwayId: depRwyId || null,
        };
        function _twNameById(id) {
          if (id == null || id === '') return null;
          const tw = (state.taxiways || []).find(function(x) { return x && x.id === id; });
          if (!tw) return String(id);
          const n = (tw.name && String(tw.name).trim()) || '';
          return n || String(tw.id || id);
        }
        function _standNameById(id) {
          if (id == null || id === '') return null;
          if (typeof findStandById === 'function') {
            const st = findStandById(id);
            if (!st) return String(id);
            const n = (st.name && String(st.name).trim()) || '';
            return n || String(st.id || id);
          }
          return String(id);
        }
        function _labelOrId(id, getLab) {
          if (id == null || id === '') return null;
          if (typeof getLab === 'function') {
            const lab = getLab(id);
            if (lab && lab !== '—') return lab;
          }
          return String(id);
        }
        copy.token_name = {
          arrRunwayId: _labelOrId(arrRwyId, typeof getRunwayDisplayLabelById === 'function' ? getRunwayDisplayLabelById : null),
          ExitTaxiwayId: exitTwId ? _twNameById(exitTwId) : null,
          apronId: apronId ? _standNameById(apronId) : null,
          terminalId: _labelOrId(termId, typeof getTerminalDisplayLabelById === 'function' ? getTerminalDisplayLabelById : null),
          depRunwayId: _labelOrId(depRwyId, typeof getRunwayDisplayLabelById === 'function' ? getRunwayDisplayLabelById : null),
        };
        const schedExport = flightScheduleMinutesForRow(f);
        copy.sibtDateTime = formatFlightScheduleDateTime(f, schedExport.sibt);
        copy.sobtDateTime = formatFlightScheduleDateTime(f, schedExport.sobt);
        copy.sldtDateTime = formatFlightScheduleDateTime(f, schedExport.sldt);
        copy.stotDateTime = formatFlightScheduleDateTime(f, schedExport.stot);
        if (state.hasSimulationResult && Array.isArray(f.timeline) && f.timeline.length >= 2) {
          copy.timeline = f.timeline.map(function(p) {
            const x = p.x != null && p.x !== '' ? Number(p.x) : Number(p.col);
            const y = p.y != null && p.y !== '' ? Number(p.y) : Number(p.row);
            const dg = p.deadlockGhost === true || p.deadlock_ghost === true;
            const o = { t: Number(p.t), x: x, y: y };
            if (dg) o.deadlockGhost = true;
            if (p.pathType != null && p.pathType !== '') o.pathType = String(p.pathType);
            if (p.phase != null && p.phase !== '') o.phase = String(p.phase);
            if (p.edgeId != null && String(p.edgeId).trim()) o.edgeId = String(p.edgeId).trim();
            return o;
          }).filter(function(k) {
            return isFinite(k.t) && isFinite(k.x) && isFinite(k.y);
          });
          if (copy.timeline.length >= 2 && f.timeline_meta && typeof f.timeline_meta === 'object') {
            try {
              copy.timeline_meta = JSON.parse(JSON.stringify(f.timeline_meta));
            } catch (eMeta) {
              copy.timeline_meta = Object.assign({}, f.timeline_meta);
            }
          }
        }
        if (state.hasSimulationResult && Array.isArray(f.proSimEdgeList) && f.proSimEdgeList.length) {
          copy.proSimEdgeList = f.proSimEdgeList.slice();
        }
        return copy;
      }),
      layoutMarkers: (state.layoutMarkers || []).map(function(m) {
        if (!m || !m.kind) return null;
        if (m.kind === 'text') {
          return { kind: 'text', id: m.id, x: Number(m.x), y: Number(m.y), text: String(m.text || '') };
        }
        if (m.kind === 'ruler') {
          return { kind: 'ruler', id: m.id, x1: Number(m.x1), y1: Number(m.y1), x2: Number(m.x2), y2: Number(m.y2) };
        }
        if (m.kind === 'island') {
          const pts = Array.isArray(m.points) ? m.points.map(function(p) {
            return { x: Number(p && p.x), y: Number(p && p.y) };
          }).filter(function(p) { return isFinite(p.x) && isFinite(p.y); }) : [];
          if (pts.length < 3) return null;
          return {
            kind: 'island',
            id: m.id,
            points: pts,
            widthM: islandWidthMResolved(m),
          };
        }
        if (m.kind === 'area') {
          const pts = Array.isArray(m.points) ? m.points.map(function(p) {
            return { x: Number(p && p.x), y: Number(p && p.y) };
          }).filter(function(p) { return isFinite(p.x) && isFinite(p.y); }) : [];
          if (pts.length < 3) return null;
          return { kind: 'area', id: m.id, points: pts };
        }
        if (m.kind === 'flight') {
          const si = (typeof m.segIndex === 'number' && isFinite(m.segIndex)) ? Math.floor(m.segIndex) : (parseInt(m.segIndex, 10) || 0);
          return {
            kind: 'flight',
            id: m.id,
            taxiwayId: m.taxiwayId,
            segIndex: si,
            t: Number(m.t),
            aircraftType: String(m.aircraftType || '').trim(),
            blazerEnabled: !!m.blazerEnabled,
            headingReversed: !!m.headingReversed,
            blazerColor: MARKER_BLAZER_COLOR_OPTIONS.indexOf(String(m.blazerColor || '').trim()) >= 0 ? String(m.blazerColor).trim() : MARKER_BLAZER_COLOR_OPTIONS[0],
            blazerLeftTrail: Array.isArray(m.blazerLeftTrail) ? m.blazerLeftTrail.map(function(p) { return { x: Number(p && p.x), y: Number(p && p.y) }; }).filter(function(p) { return isFinite(p.x) && isFinite(p.y); }) : [],
            blazerRightTrail: Array.isArray(m.blazerRightTrail) ? m.blazerRightTrail.map(function(p) { return { x: Number(p && p.x), y: Number(p && p.y) }; }).filter(function(p) { return isFinite(p.x) && isFinite(p.y); }) : []
          };
        }
        if (m.kind === 'navaid') {
          return { kind: 'navaid', id: m.id, subType: (m.subType === 'ils') ? 'ils' : 'papi', x: Number(m.x), y: Number(m.y) };
        }
        return null;
      }).filter(Boolean),
      designerPersist: {
        v: 1,
        globalUpdateFresh: !!state.globalUpdateFresh,
        designerPageUpdateFresh: !!state.designerPageUpdateFresh,
        hasSimulationPlayback: !!state.hasSimulationResult,
        simPlaybackEndCapSec: (state.simPlaybackEndCapSec != null && isFinite(Number(state.simPlaybackEndCapSec)))
          ? Number(state.simPlaybackEndCapSec)
          : null,
        mapTypeMode: (state.mapTypeMode === 'heatmap') ? 'heatmap' : 'normal',
        heatmapTrafficPhases: Object.assign({}, state.heatmapTrafficPhases || {}),
      },
      simPathGraph: buildSimPathGraphExport()
    };
  }
  function buildLayout3DViewerPayload() {
    let tSec = Number(state.simTimeSec);
    if (!isFinite(tSec)) tSec = 0;
    const layout = serializeCurrentLayout();
    const flightDrawPoses = [];
    (state.flights || []).forEach(function(f) {
      if (!f) return;
      let pose = null;
      if (typeof getFlightPoseAtTimeForDraw === 'function') {
        pose = getFlightPoseAtTimeForDraw(f, tSec);
      }
      flightDrawPoses.push({
        id: f.id,
        reg: f.reg,
        aircraftType: f.aircraftType,
        code: f.code,
        arrDep: f.arrDep,
        pose: pose && isFinite(pose.x) && isFinite(pose.y) ? { x: pose.x, y: pose.y, dx: pose.dx, dy: pose.dy } : null
      });
    });
    const enrichedFootprints = {
      remote: (state.remoteStands || []).map(function(st) {
        return {
          id: st && st.id,
          name: st && st.name,
          corners: typeof getRemoteStandCorners === 'function' ? getRemoteStandCorners(st) : null
        };
      }).filter(function(r) { return r.corners && r.corners.length >= 3; }),
      pbb: (state.pbbStands || []).map(function(pbb) {
        return {
          id: pbb && pbb.id,
          name: pbb && pbb.name,
          corners: typeof getPBBStandCorners === 'function' ? getPBBStandCorners(pbb) : null
        };
      }).filter(function(r) { return r.corners && r.corners.length >= 3; })
    };
    const enrichedApronLinkPolylines = (state.apronLinks || []).map(function(lk) {
      if (!lk || typeof getApronLinkPolylineWorldPts !== 'function') return null;
      const pts = getApronLinkPolylineWorldPts(lk);
      if (!pts || pts.length < 2) return null;
      return {
        id: lk.id,
        points: pts.map(function(p) { return { x: p[0], y: p[1] }; })
      };
    }).filter(Boolean);
    const payload = {
      version: 1,
      kind: 'grid3dViewer',
      layoutApiUrl: (typeof LAYOUT_API_URL === 'string' && LAYOUT_API_URL) ? LAYOUT_API_URL : '',
      grid3dAssetApiUrl: (typeof GRID3D_ASSET_API_URL === 'string' && GRID3D_ASSET_API_URL) ? GRID3D_ASSET_API_URL : '',
      exportedAt: new Date().toISOString(),
      simTimeSec: tSec,
      viewerConfig: {
        gridMajorInterval: GRID_MAJOR_INTERVAL,
        gridViewBg: GRID_VIEW_BG
      },
      layout: layout,
      flightDrawPoses: flightDrawPoses,
      enrichedFootprints: enrichedFootprints,
      enrichedApronLinkPolylines: enrichedApronLinkPolylines
    };
    try {
      let tiled = null;
      if (typeof exportLayoutGroundTilesFor3D === 'function') tiled = exportLayoutGroundTilesFor3D();
      if (tiled && tiled.tiles && tiled.tiles.length === 4) {
        payload.layoutGroundTiles = tiled;
      } else if (typeof exportLayoutGroundTextureFor3D === 'function') {
        const gt = exportLayoutGroundTextureFor3D();
        if (gt && gt.dataUrl) payload.layoutGroundTexture = gt;
      }
    } catch (eTex) {
      console.warn('exportLayoutGroundTilesFor3D / exportLayoutGroundTextureFor3D failed', eTex);
    }
    return payload;
  }
  function openGrid3DViewerWindow() {
    const tpl = typeof window.__GRID3D_VIEWER_HTML_TEMPLATE__ === 'string' ? window.__GRID3D_VIEWER_HTML_TEMPLATE__ : '';
    if (!tpl || tpl.length < 80) {
      console.error('Grid 3D viewer template missing');
      alert('3D viewer template is not loaded. Ensure pages/Layout_Design/3D/grid3d-viewer.html exists and reload the Layout Design page.');
      return;
    }
    const bootHtml = '<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/><title>Layout 3D</title>' +
      '<style>html,body{margin:0;height:100%;background:#0d0d0f;color:#e2e8f0;font-family:system-ui,sans-serif;overflow:hidden}' +
      '.wrap{display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:100vh;gap:18px;padding:24px;box-sizing:border-box}' +
      '.sp{width:44px;height:44px;border:3px solid rgba(148,163,184,.25);border-top-color:#7c6af7;border-radius:50%;animation:g .85s linear infinite}' +
      '@keyframes g{to{transform:rotate(360deg)}}' +
      '.bar{width:min(360px,86vw);height:4px;border-radius:2px;background:rgba(148,163,184,.2);overflow:hidden}' +
      '.bar>i{display:block;height:100%;width:38%;background:linear-gradient(90deg,#5b52d6,#7c6af7);border-radius:2px;animation:p 1.15s ease-in-out infinite}' +
      '@keyframes p{0%,100%{transform:translateX(-40%)}50%{transform:translateX(200%)}}' +
      '.t{font-size:15px;font-weight:600;color:#f1f5f9;text-align:center}.s{font-size:13px;color:#94a3b8;text-align:center;max-width:360px;line-height:1.45}' +
      '</style></head><body><div class="wrap"><div class="sp"></div><div class="bar"><i></i></div><p class="t">3D 뷰 준비 중</p>' +
      '<p class="s">레이아웃 스냅샷을 만들고 있습니다. 잠시만 기다려 주세요.</p></div></body></html>';
    const g3Base = (typeof GRID3D_ASSET_API_URL === 'string' && GRID3D_ASSET_API_URL.trim()) ? GRID3D_ASSET_API_URL.trim() : '';
    const viewerShellUrl = /^https?:\/\//i.test(g3Base) ? g3Base.replace(/\/$/, '') + '/api/grid3d-viewer-app' : '';
    let w = null;
    let openedViaReceiverShell = false;
    if (viewerShellUrl) {
      try {
        w = window.open(viewerShellUrl, '_blank', 'width=1280,height=840');
        openedViaReceiverShell = !!w;
      } catch (eHttp) {
        console.warn('Grid 3D receiver shell open failed', eHttp);
        w = null;
        openedViaReceiverShell = false;
      }
    }
    if (!w) {
      try {
        w = window.open('data:text/html;charset=utf-8,' + encodeURIComponent(bootHtml), '_blank', 'width=1280,height=840');
      } catch (eData) {
        console.warn('Grid 3D popup data URL failed, using about:blank', eData);
      }
    }
    if (!w) {
      w = window.open('about:blank', '_blank', 'width=1280,height=840');
    }
    if (!w) {
      alert('Popup was blocked. Allow popups for this site to open the 3D viewer.');
      return;
    }
    if (!openedViaReceiverShell) {
      var bootHref = '';
      try {
        bootHref = w.location && w.location.href ? String(w.location.href) : '';
      } catch (eLoc) {
        bootHref = '';
      }
      if (bootHref.indexOf('data:') !== 0) {
        try {
          w.document.open();
          w.document.write(bootHtml);
          w.document.close();
        } catch (eOpen) {
          console.error(eOpen);
          try {
            w.close();
          } catch (eClose) { /* ignore */ }
          alert('Could not open the 3D viewer window.');
          return;
        }
      }
    }
    let payload;
    try {
      payload = typeof buildLayout3DViewerPayload === 'function' ? buildLayout3DViewerPayload() : null;
    } catch (e) {
      console.error('buildLayout3DViewerPayload failed:', e);
      try {
        w.close();
      } catch (eClose2) { /* ignore */ }
      alert('Could not serialize layout for 3D: ' + (e && e.message ? e.message : e));
      return;
    }
    if (!payload || !payload.layout) {
      try {
        w.close();
      } catch (eClose3) { /* ignore */ }
      alert('Could not serialize layout for 3D.');
      return;
    }
    function sendGrid3dInit() {
      try {
        w.postMessage({ kind: 'grid3dViewerInit', payload: payload }, '*');
      } catch (e4) {
        console.error('postMessage to 3D viewer failed:', e4);
        alert('Could not send layout data to the 3D window. Try again or check the browser console.');
      }
    }
    if (!openedViaReceiverShell) {
      try {
        w.document.open();
        w.document.write(tpl);
        w.document.close();
      } catch (e3) {
        console.error(e3);
        try {
          w.close();
        } catch (eClose4) { /* ignore */ }
        alert('Could not write the 3D viewer document.');
        return;
      }
      setTimeout(sendGrid3dInit, 0);
    } else {
      function onShellReady() {
        setTimeout(sendGrid3dInit, 0);
      }
      try {
        if (w.document && w.document.readyState === 'complete') {
          onShellReady();
        } else {
          w.addEventListener('load', function grid3dShellLoad() {
            w.removeEventListener('load', grid3dShellLoad);
            onShellReady();
          });
        }
      } catch (eReady) {
        setTimeout(sendGrid3dInit, 150);
      }
    }
  }
  function getExistingStandBounds() {
    const list = [];
    state.remoteStands.forEach(st => {
      const corners = getRemoteStandCorners(st);
      let left = corners[0][0], right = corners[0][0], top = corners[0][1], bottom = corners[0][1];
      for (let k = 1; k < 4; k++) {
        left = Math.min(left, corners[k][0]); right = Math.max(right, corners[k][0]);
        top = Math.min(top, corners[k][1]); bottom = Math.max(bottom, corners[k][1]);
      }
      list.push({ left, right, top, bottom });
    });
    state.pbbStands.forEach(pbb => {
      const corners = getPBBStandCorners(pbb);
      let left = corners[0][0], right = corners[0][0], top = corners[0][1], bottom = corners[0][1];
      for (let k = 1; k < 4; k++) {
        left = Math.min(left, corners[k][0]); right = Math.max(right, corners[k][0]);
        top = Math.min(top, corners[k][1]); bottom = Math.max(bottom, corners[k][1]);
      }
      list.push({ left, right, top, bottom });
    });
    return list;
  }
  function standOverlapsExisting(bounds) {
    const existing = getExistingStandBounds();
    for (let i = 0; i < existing.length; i++) if (rectsOverlap(bounds, existing[i])) return true;
    return false;
  }
  function dist2(a, b) { const dx = a[0]-b[0], dy = a[1]-b[1]; return dx*dx+dy*dy; }
  function _normalizeTimeToSeconds(value, unit, roundingMode) {
    const raw = Number(value || 0);
    const scaled = unit === 'minutes' ? raw * 60 : raw;
    const rounded = roundingMode === 'round' ? Math.round(scaled) : Math.floor(scaled);
    return Math.max(0, rounded);
  }
  function _splitTotalSeconds(totalSec) {
    const safeSec = Math.max(0, Math.floor(totalSec || 0));
    const h = Math.floor(safeSec / 3600);
    const m = Math.floor((safeSec % 3600) / 60);
    const s = safeSec % 60;
    return {
      h,
      m,
      s,
      hh: (h < 10 ? '0' : '') + h,
      mm: (m < 10 ? '0' : '') + m,
      ss: (s < 10 ? '0' : '') + s,
    };
  }
  function formatMinutesToHHMM(m) {
    const parts = _splitTotalSeconds(_normalizeTimeToSeconds(m, 'minutes', 'floor'));
    return parts.h + ':' + parts.mm;
  }
  function findNearestItem(candidates, getPoint, wx, wy, maxD2) {
    const click = [wx, wy];
    let best = null;
    let bestD2 = maxD2;
    for (let i = 0; i < candidates.length; i++) {
      const c = candidates[i];
      const pt = getPoint(c);
      if (!pt || pt.length < 2) continue;
      const d2 = dist2(pt, click);
      if (d2 < bestD2) {
        bestD2 = d2;
        best = c;
      }
    }
    return best;
  }
  function closestPointOnSegment(p1, p2, p) {
    const [x1,y1]=p1,[x2,y2]=p2,[px,py]=p;
    const dx=x2-x1,dy=y2-y1,len2=dx*dx+dy*dy;
    if (len2===0) return null;
    let t = ((px-x1)*dx+(py-y1)*dy)/len2;
    t = Math.max(0,Math.min(1,t));
    return [x1+t*dx,y1+t*dy];
  }
  function getClosestTerminalEdgePoint(wx, wy) {
    const click = [wx, wy];
    let best = null;
    let bestD2 = Infinity;
    (state.terminals || []).forEach(function(term) {
      if (!term || !term.closed || !Array.isArray(term.vertices) || term.vertices.length < 2) return;
      for (let i = 0; i < term.vertices.length; i++) {
        const v1 = term.vertices[i];
        const v2 = term.vertices[(i + 1) % term.vertices.length];
        const p1 = cellToPixel(v1.col, v1.row);
        const p2 = cellToPixel(v2.col, v2.row);
        const near = closestPointOnSegment(p1, p2, click);
        if (!near) continue;
        const d2 = dist2(near, click);
        if (d2 < bestD2) {
          bestD2 = d2;
          best = { point: near, term: term, edgeIndex: i };
        }
      }
    });
    return best;
  }
  function getPbbBoardingWidthM(pbb) {
    const w = Number(pbb && pbb.boardingWidthM);
    if (isFinite(w) && w > 0) return w;
    return 5;
  }
  function getPbbBoardingHeightM(pbb) {
    const h = Number(pbb && pbb.boardingHeightM);
    if (isFinite(h) && h > 0) return h;
    return 15;
  }
  function getPbbTerminalContactSetbackM(pbb) {
    const v = Number(pbb && pbb.terminalContactSetbackM);
    if (isFinite(v) && v >= 0) return v;
    return 0;
  }
  function getPbbTerminalFrameFromEdge(term, edgeIndex, wallX, wallY) {
    const v1 = term.vertices[edgeIndex], v2 = term.vertices[(edgeIndex + 1) % term.vertices.length];
    const p1 = cellToPixel(v1.col, v1.row), p2 = cellToPixel(v2.col, v2.row);
    const edx = p2[0] - p1[0], edy = p2[1] - p1[1];
    const el = Math.hypot(edx, edy) || 1;
    const tx = edx / el, ty = edy / el;
    let nx = -ty, ny = tx;
    let tcx = 0, tcy = 0;
    term.vertices.forEach(function(v) {
      const q = cellToPixel(v.col, v.row);
      tcx += q[0];
      tcy += q[1];
    });
    tcx /= term.vertices.length;
    tcy /= term.vertices.length;
    const inX = tcx - wallX, inY = tcy - wallY;
    if (nx * inX + ny * inY > 0) {
      nx = -nx;
      ny = -ny;
    }
    return { tx: tx, ty: ty, nx: nx, ny: ny };
  }
  function getPbbTerminalFrameAtWorld(wx, wy) {
    const proj = getClosestTerminalEdgePoint(wx, wy);
    if (!proj || !proj.term) return null;
    const fr = getPbbTerminalFrameFromEdge(proj.term, proj.edgeIndex, proj.point[0], proj.point[1]);
    return { mpx: proj.point[0], mpy: proj.point[1], term: proj.term, edgeIndex: proj.edgeIndex, tx: fr.tx, ty: fr.ty, nx: fr.nx, ny: fr.ny };
  }
  function ensurePbbBoardingWallGeometry(pbb) {
    if (!pbb || !Array.isArray(pbb.pbbBridges) || !pbb.pbbBridges.length) return;
    const Tsrcx = Number.isFinite(Number(pbb.x1)) ? Number(pbb.x1) : Number(pbb.pbbBridges[0].points[0].x) || 0;
    const Tsrcy = Number.isFinite(Number(pbb.y1)) ? Number(pbb.y1) : Number(pbb.pbbBridges[0].points[0].y) || 0;
    const proj = getClosestTerminalEdgePoint(Tsrcx, Tsrcy);
    if (!proj || !proj.term) return;
    const fr = getPbbTerminalFrameFromEdge(proj.term, proj.edgeIndex, proj.point[0], proj.point[1]);
    const wx = proj.point[0], wy = proj.point[1];
    const Tx = wx, Ty = wy;
    const depthM = getPbbBoardingHeightM(pbb);
    const Bx = Tx + fr.nx * depthM, By = Ty + fr.ny * depthM;
    pbb.pbbBridges.forEach(function(bridge) {
      if (!bridge.points || bridge.points.length < 3) return;
      bridge.points[0].x = Tx;
      bridge.points[0].y = Ty;
      bridge.points[1].x = Bx;
      bridge.points[1].y = By;
    });
    pbb.x1 = Tx;
    pbb.y1 = Ty;
    pbb.x2 = Bx;
    pbb.y2 = By;
  }
  function applyPbbArmLengthToBridgeEnds(pbb, armLenM) {
    if (!pbb || !Array.isArray(pbb.pbbBridges)) return;
    ensurePbbBoardingWallGeometry(pbb);
    const len = Math.max(3, Number(armLenM) || 15);
    pbb.pbbBridges.forEach(function(bridge) {
      const pts = bridge.points;
      if (!pts || pts.length < 3) return;
      const bx = Number(pts[1].x), by = Number(pts[1].y);
      const px = Number(pts[2].x), py = Number(pts[2].y);
      let vx = px - bx, vy = py - by;
      const hl = Math.hypot(vx, vy) || 1;
      vx /= hl;
      vy /= hl;
      pts[2].x = bx + vx * len;
      pts[2].y = by + vy * len;
    });
    bumpPathPolylineCacheRev();
  }
  function getPbbBoardingRectangleCornersWorldPx(pbb) {
    ensurePbbBoardingWallGeometry(pbb);
    const proj = getClosestTerminalEdgePoint(Number(pbb.x1) || 0, Number(pbb.y1) || 0);
    if (!proj || !proj.term) return null;
    const fr = getPbbTerminalFrameFromEdge(proj.term, proj.edgeIndex, proj.point[0], proj.point[1]);
    const wx = proj.point[0], wy = proj.point[1];
    const Tx = wx, Ty = wy;
    const halfW = getPbbBoardingWidthM(pbb) * 0.5;
    const depthM = getPbbBoardingHeightM(pbb);
    const c0 = [Tx - fr.tx * halfW, Ty - fr.ty * halfW];
    const c1 = [Tx + fr.tx * halfW, Ty + fr.ty * halfW];
    const c2 = [c1[0] + fr.nx * depthM, c1[1] + fr.ny * depthM];
    const c3 = [c0[0] + fr.nx * depthM, c0[1] + fr.ny * depthM];
    return [c0, c1, c2, c3];
  }
  function drawPbbBoardingRectangle(ctx, pbb, sel) {
    const poly = getPbbBoardingRectangleCornersWorldPx(pbb);
    if (!poly || poly.length < 4) return;
    const nowPerf = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
    const suppressStandFill = !!state.isPanning || nowPerf < _layoutDetailSuppressUntil;
    const sl = !!state.layers.standLines, sf = !!state.layers.standFill && !suppressStandFill;
    const monoFillP = layerMonoFillOn() && !sel;
    const monoLineP = layerMonoLinesOn() && !sel;
    const monoLp = c2dLayerMonoLineStrokeCss();
    if (!sl && !sf) return;
    ctx.save();
