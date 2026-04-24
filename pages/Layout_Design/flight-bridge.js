    ctx.lineTo(xStop + shiftX, noseHalf);
    ctx.lineTo(Math.min(xBendEnd, halfD) + shiftX, halfW);
    if (xBendEnd < halfD - eps) {
      ctx.lineTo(halfD + shiftX, halfW);
      ctx.lineTo(halfD + shiftX, -halfW);
      ctx.lineTo(xBendEnd + shiftX, -halfW);
    } else {
      ctx.lineTo(halfD + shiftX, -halfW);
    }
    ctx.lineTo(xStop + shiftX, -noseHalf);
    ctx.closePath();
    return true;
  }
  /** Stand-local +X = tail/apron-open, −X = nose/terminal-ward. Red dashed: stop bar (nose_clear), pushback (pushback), lateral gap bounds (wingspan ± vs stand width). Clipped to safety footprint when nose geometry applies. */
  function drawStandApronMarkingsInLocalAxes(ctx, depM, widM, category) {
    const r = standConfigRowForIcaoCat(category);
    if (!r || !isFinite(depM) || !isFinite(widM) || depM <= 0 || widM <= 0) return;
    const halfD = depM / 2, halfW = widM / 2;
    const eps = 0.12;
    ctx.save();
    if (buildStandSafetyPolygonPath(ctx, depM, widM, category)) {
      ctx.clip();
    }
    ctx.strokeStyle = layerMonoLinesOn() ? c2dLayerMonoLineStrokeCss() : c2dStandSafetyStroke();
    ctx.lineWidth = Math.max(0.35, 0.42 / Math.max(state.scale, 0.1));
    ctx.setLineDash([2, 2.5]);
    ctx.lineCap = 'butt';
    ctx.lineJoin = 'miter';
    const nc = Number(r.nose_clear), pb = Number(r.pushback);
    const shiftX = standStopbarCenterShiftLocalX(depM, category);
    if (isFinite(nc) && isFinite(pb)) {
      const xStop = 0;
      const xPush = (halfD - pb) + shiftX;
      const xMin = (-halfD) + shiftX;
      const xMax = (halfD) + shiftX;
      if (xStop > -halfD + eps && xStop < halfD - eps) {
        ctx.beginPath();
        ctx.moveTo(xStop, -halfW);
        ctx.lineTo(xStop, halfW);
        ctx.stroke();
      }
      if (xPush < xMax - eps && xPush > xMin + eps) {
        ctx.beginPath();
        ctx.moveTo(xPush, -halfW);
        ctx.lineTo(xPush, halfW);
        ctx.stroke();
      }
    }
    const g = Number(r.gap), ws = Number(r.wingspan);
    if (isFinite(g) && g > eps && isFinite(ws) && ws > 0) {
      const yLim = halfW - g;
      if (yLim > eps && yLim < halfW - eps) {
        ctx.beginPath();
        ctx.moveTo(-halfD + shiftX, yLim);
        ctx.lineTo(halfD + shiftX, yLim);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(-halfD + shiftX, -yLim);
        ctx.lineTo(halfD + shiftX, -yLim);
        ctx.stroke();
      }
    }
    ctx.restore();
  }
  function fillStandSafetyFootprintInLocalAxes(ctx, depM, widM, category) {
    if (buildStandSafetyPolygonPath(ctx, depM, widM, category)) {
      ctx.fill();
      return;
    }
    const shiftX = standStopbarCenterShiftLocalX(depM, category);
    ctx.beginPath();
    ctx.rect((-depM / 2) + shiftX, -widM / 2, depM, widM);
    ctx.fill();
  }
  function drawStandSafetyContourInLocalAxes(ctx, depM, widM, category, selected) {
    if (!buildStandSafetyPolygonPath(ctx, depM, widM, category)) return;
    ctx.save();
    ctx.strokeStyle = (!selected && layerMonoLinesOn()) ? c2dLayerMonoLineStrokeCss() : c2dStandSafetyStroke();
    const baseLw = Math.max(0.55, 0.65 / Math.max(state.scale, 0.1));
    ctx.lineWidth = selected ? baseLw * 1.35 : baseLw;
    ctx.setLineDash([]);
    ctx.lineJoin = 'miter';
    ctx.lineCap = 'butt';
    if (selected) {
      ctx.shadowColor = c2dObjectSelectedGlow();
      ctx.shadowBlur = c2dObjectSelectedGlowBlur();
    }
    ctx.stroke();
    ctx.restore();
  }
  /** Local +X from stand box center to end of 45° flare (full-width main body); 0 if nose geometry unused. */
  function standSafetyAircraftCenterLocalXM(depM, widM, category) {
    const r = standConfigRowForIcaoCat(category);
    if (!r || !isFinite(depM) || !isFinite(widM) || depM <= 0 || widM <= 0) return 0;
    const nw = Number(r.nose_width), nc = Number(r.nose_clear);
    if (!isFinite(nw) || nw <= 0 || !isFinite(nc) || nc <= 0) return 0;
    const halfD = depM / 2, halfW = widM / 2;
    const noseHalf = nw / 2;
    const eps = 0.08;
    if (noseHalf >= halfW - eps) return 0;
    const xNose = -halfD;
    const xStop = -halfD + nc;
    if (xStop <= xNose + eps || xStop >= halfD - eps) return 0;
    const latRun = halfW - noseHalf;
    if (latRun <= eps) return 0;
    const xBendEnd = xStop + latRun;
    if (xBendEnd > halfD + eps) return 0;
    return xBendEnd;
  }
  function getStandAircraftMarkerWorldPxForPbb(pbb) {
    const cxy = getStandConnectionPx(pbb);
    return [cxy[0], cxy[1]];
  }
  function getStandAircraftMarkerWorldPxForRemoteLike(st) {
    const cxy = getStandConnectionPx(st);
    return [cxy[0], cxy[1]];
  }
  const APRON_SITE_MARKER_WIDTH_M = 5;
  const APRON_SITE_MARKER_HEIGHT_M = 1;
  function _ensureLayoutToastStack() {
    let stack = document.getElementById('layout-toast-stack');
    if (!stack) {
      stack = document.createElement('div');
      stack.id = 'layout-toast-stack';
      stack.className = 'layout-toast-stack';
      document.body.appendChild(stack);
    }
    return stack;
  }
  function _formatToastTimestamp(d) {
    const pad = function(n) { return String(n).padStart(2, '0'); };
    return pad(d.getHours()) + ':' + pad(d.getMinutes()) + ':' + pad(d.getSeconds());
  }
  function showLayoutSavedToast(layoutName, kind, subText) {
    const stack = _ensureLayoutToastStack();
    const el = document.createElement('div');
    const variant = kind === 'error' ? 'is-error' : 'is-success';
    el.className = 'layout-toast ' + variant;
    const title = document.createElement('div');
    title.className = 'layout-toast-title';
    const ts = _formatToastTimestamp(new Date());
    const name = String(layoutName || '').trim() || 'layout';
    if (kind === 'error') {
      title.textContent = ts + ' · save failed · ' + name;
    } else {
      title.textContent = ts + ' · saved · ' + name;
    }
    el.appendChild(title);
    if (subText) {
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
