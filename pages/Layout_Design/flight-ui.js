    if (!raw || typeof raw !== 'object') return out;
    Object.keys(raw).forEach(function(k) {
      if (k === 'description') return;
      const row = raw[k];
      if (!row || typeof row !== 'object') return;
      const ws = Number(row.wingspan), g = Number(row.gap), rd = Number(row.road);
      const ln = Number(row.length), nc = Number(row.nose_clear), pb = Number(row.pushback);
      const dp = Number(row.depth);
      if (![ws, g, rd, ln, nc, pb, dp].every(isFinite)) return;
      const rowOut = { wingspan: ws, gap: g, road: rd, length: ln, nose_clear: nc, pushback: pb, depth: dp };
      const og = Number(row.outer_gear), nsc = Number(row.nose_side_clear), nw = Number(row.nose_width);
      if (isFinite(og)) rowOut.outer_gear = og;
      if (isFinite(nsc)) rowOut.nose_side_clear = nsc;
      if (isFinite(nw)) rowOut.nose_width = nw;
      out[String(k).toUpperCase().slice(0, 1)] = rowOut;
    });
    return out;
  })();
  function standConfigRowForIcaoCat(cat) {
    const s = String(cat == null ? 'C' : cat).toUpperCase();
    let c = 'C';
    for (let i = 0; i < s.length; i++) {
      const ch = s.charAt(i);
      if (ch >= 'A' && ch <= 'F') {
        c = ch;
        break;
      }
    }
    return STAND_CONFIG_ROW_BY_CODE[c] || null;
  }
  function getStandWidthMeters(cat) {
    const r = standConfigRowForIcaoCat(cat);
    if (r) return r.wingspan + r.gap * 2;
    return getStandSizeMeters(cat);
  }
  function getStandDepthMeters(cat) {
    const r = standConfigRowForIcaoCat(cat);
    if (r) return r.depth;
    return getStandSizeMeters(cat);
  }
  function getStandSpacingMeters(catA, catB) {
    const ra = standConfigRowForIcaoCat(catA);
    const rb = standConfigRowForIcaoCat(catB);
    if (ra && rb) return Math.max(ra.road, rb.road);
    return 0;
  }
  function standFootprintLocalToWorld(cx, cy, angleRad, lx, ly) {
    const cos = Math.cos(angleRad), sin = Math.sin(angleRad);
    return [cx + lx * cos - ly * sin, cy + lx * sin + ly * cos];
  }
  function standStopbarCenterShiftLocalX(depM, category) {
    const r = standConfigRowForIcaoCat(category);
    if (!r || !isFinite(depM) || depM <= 0) return 0;
    const nc = Number(r.nose_clear);
    if (!isFinite(nc) || nc <= 0) return 0;
    const halfD = depM / 2;
    const xStopOld = -halfD + nc;
    if (!(xStopOld > -halfD && xStopOld < halfD)) return 0;
    return -xStopOld;
  }
  /** Nose (−X) width = nose_width; stop bar at nose_clear from nose edge; 45° flare to full stand width (±halfW), then rectangle to tail (+halfD). */
  function buildStandSafetyPolygonPath(ctx, depM, widM, category) {
    const r = standConfigRowForIcaoCat(category);
    if (!r || !isFinite(depM) || !isFinite(widM) || depM <= 0 || widM <= 0) return false;
    const nw = Number(r.nose_width), nc = Number(r.nose_clear);
    if (!isFinite(nw) || nw <= 0 || !isFinite(nc) || nc <= 0) return false;
    const halfD = depM / 2, halfW = widM / 2;
    const noseHalf = nw / 2;
    const eps = 0.08;
    if (noseHalf >= halfW - eps) return false;
    const xNose = -halfD;
    const xStop = -halfD + nc;
    if (xStop <= xNose + eps || xStop >= halfD - eps) return false;
    const latRun = halfW - noseHalf;
    if (latRun <= eps) return false;
    const xBendEnd = xStop + latRun;
    if (xBendEnd > halfD + eps) return false;
    const shiftX = standStopbarCenterShiftLocalX(depM, category);
    ctx.beginPath();
    ctx.moveTo(xNose + shiftX, -noseHalf);
    ctx.lineTo(xNose + shiftX, noseHalf);
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
  function _svgLayoutToastCheckIcon() {
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('class', 'layout-toast__icon');
    svg.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
    svg.setAttribute('viewBox', '0 0 20 20');
    svg.setAttribute('fill', 'currentColor');
    svg.setAttribute('aria-hidden', 'true');
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('fill-rule', 'evenodd');
    path.setAttribute('d', 'M10 18a8 8 0 100-16 8 8 0 000 16zm3.857-9.809a.75.75 0 00-1.214-.882l-3.483 4.79-1.88-1.88a.75.75 0 10-1.06 1.061l2.5 2.5a.75.75 0 001.137-.089l4-5.5z');
    path.setAttribute('clip-rule', 'evenodd');
    svg.appendChild(path);
    return svg;
  }
  function _svgLayoutToastWarnIcon() {
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('class', 'layout-toast__icon');
    svg.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
    svg.setAttribute('viewBox', '0 0 20 20');
    svg.setAttribute('fill', 'currentColor');
    svg.setAttribute('aria-hidden', 'true');
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('fill-rule', 'evenodd');
    path.setAttribute('d', 'M8.485 2.495c.673-1.167 2.357-1.167 3.03 0l6.28 10.875c.673 1.167-.17 2.63-1.516 2.63H3.72c-1.347 0-2.189-1.463-1.515-2.63L8.485 2.495zM10 5a.75.75 0 01.75.75v3.5a.75.75 0 01-1.5 0v-3.5A.75.75 0 0110 5zm0 9a1 1 0 100-2 1 1 0 000 2z');
    path.setAttribute('clip-rule', 'evenodd');
    svg.appendChild(path);
    return svg;
  }
  function showLayoutSavedToast(layoutName, kind, subText) {
    const stack = _ensureLayoutToastStack();
    const el = document.createElement('div');
    const variant = kind === 'error' ? 'is-error' : 'is-success';
    el.className = 'layout-toast ' + variant;
    el.setAttribute('role', kind === 'error' ? 'alert' : 'status');
    const icon = kind === 'error' ? _svgLayoutToastWarnIcon() : _svgLayoutToastCheckIcon();
    const content = document.createElement('div');
    content.className = 'layout-toast__content';
    const titleEl = document.createElement('div');
    titleEl.className = 'layout-toast__title';
    titleEl.textContent = kind === 'error' ? 'Save failed' : 'Saved';
    const textEl = document.createElement('span');
    textEl.className = 'layout-toast__text';
    const ts = _formatToastTimestamp(new Date());
    const name = String(layoutName || '').trim() || 'layout';
    var body = ts + ' · ' + name;
    if (kind !== 'error' && !name.endsWith('.json')) {
      body = body + '.json';
    }
    if (subText) {
      body = body + '\n' + String(subText);
    }
    textEl.textContent = body;
    content.appendChild(titleEl);
    content.appendChild(textEl);
    el.appendChild(icon);
    el.appendChild(content);
    stack.appendChild(el);
    requestAnimationFrame(function() { el.classList.add('is-visible'); });
    const lifeMs = kind === 'error' ? 4200 : 2600;
    setTimeout(function() {
      el.classList.remove('is-visible');
      setTimeout(function() { if (el.parentNode === stack) stack.removeChild(el); }, 220);
    }, lifeMs);
  }
  /** Same visuals as Save success toast; custom title/detail (no `.json` suffix). */
  function showDesignerSuccessToast(title, detailLine) {
    const stack = _ensureLayoutToastStack();
    const el = document.createElement('div');
    el.className = 'layout-toast is-success';
    el.setAttribute('role', 'status');
    const icon = _svgLayoutToastCheckIcon();
    const content = document.createElement('div');
    content.className = 'layout-toast__content';
    const titleEl = document.createElement('div');
    titleEl.className = 'layout-toast__title';
    titleEl.textContent = String(title || '').trim() || 'Done';
    const textEl = document.createElement('span');
    textEl.className = 'layout-toast__text';
    const ts = _formatToastTimestamp(new Date());
    textEl.textContent = detailLine != null && String(detailLine).trim() !== '' ? ts + ' · ' + String(detailLine).trim() : ts;
    content.appendChild(titleEl);
    content.appendChild(textEl);
    el.appendChild(icon);
    el.appendChild(content);
    stack.appendChild(el);
    requestAnimationFrame(function() { el.classList.add('is-visible'); });
    const lifeMs = 2600;
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
  function polygonsOverlapXY(polyA, polyB) {
    if (!Array.isArray(polyA) || !Array.isArray(polyB) || polyA.length < 3 || polyB.length < 3) return false;
    for (let i = 0; i < polyA.length; i++) if (pointInPolygonXY(polyA[i], polyB)) return true;
    for (let i = 0; i < polyB.length; i++) if (pointInPolygonXY(polyB[i], polyA)) return true;
    for (let i = 0; i < polyA.length; i++) {
      const a1 = polyA[i], a2 = polyA[(i + 1) % polyA.length];
      for (let j = 0; j < polyB.length; j++) {
        if (segIntersect(a1, a2, polyB[j], polyB[(j + 1) % polyB.length])) return true;
      }
    }
    return false;
  }
  function polygonAabbXY(poly) {
    if (!Array.isArray(poly) || !poly.length) return null;
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (let i = 0; i < poly.length; i++) {
      const p = poly[i];
      const x = Number(p && p[0]), y = Number(p && p[1]);
      if (!isFinite(x) || !isFinite(y)) continue;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
    return isFinite(minX) && isFinite(minY) && isFinite(maxX) && isFinite(maxY)
      ? { left: minX, right: maxX, top: minY, bottom: maxY }
      : null;
  }
  function aabbRectOverlap(a, b) {
    return !!(a && b && !(a.right < b.left || a.left > b.right || a.bottom < b.top || a.top > b.bottom));
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
  function buildStandDuplicateSafetyPolygonLocalPoints(depM, widM, category) {
    const r = standConfigRowForIcaoCat(category);
    if (!r || !isFinite(depM) || !isFinite(widM) || depM <= 0 || widM <= 0) return null;
    const g = Number(r.gap), ws = Number(r.wingspan), pb = Number(r.pushback);
    const nw = Number(r.nose_width), nc = Number(r.nose_clear);
    if (!isFinite(g) || !isFinite(ws) || !isFinite(pb) || !isFinite(nw) || !isFinite(nc) || g <= 0 || ws <= 0 || nw <= 0 || nc <= 0) return null;
    const halfD = depM / 2, halfW = widM / 2;
    const noseHalf = nw / 2;
    const shiftX = standStopbarCenterShiftLocalX(depM, category);
    const xNose = -halfD + shiftX;
    const xStop = 0;
    const xBendEnd = xStop + (halfW - noseHalf);
    const xPush = (halfD - pb) + shiftX;
    const yLim = halfW - g;
    const xMin = (-halfD) + shiftX;
    const xMax = halfD + shiftX;
    const eps = 0.12;
    if (!(noseHalf < halfW - eps)) return null;
    if (!(xBendEnd <= xMax + eps)) return null;
    if (!(yLim > eps && yLim < halfW - eps)) return null;
    if (!(xStop > xMin + eps && xStop < xMax - eps)) return null;
    const xA = Math.max(xMin, Math.min(xMax, Math.min(xStop, xPush)));
    const xB = Math.max(xMin, Math.min(xMax, Math.max(xStop, xPush)));
    if (!(xB > xA + eps)) return null;
    const contour = [
      [xNose, -noseHalf],
      [xNose, noseHalf],
      [xStop, noseHalf],
      [Math.min(xBendEnd, xMax), halfW],
      [xMax, halfW],
      [xMax, -halfW],
      [Math.min(xBendEnd, xMax), -halfW],
      [xStop, -noseHalf],
    ];
    return clipPolygonToAxisAlignedBox(contour, xA, xB, -yLim, yLim);
  }
  function clipPolygonToAxisAlignedBox(poly, minX, maxX, minY, maxY) {
    if (!Array.isArray(poly) || poly.length < 3) return null;
    function clip(input, axis, keepGreater, value) {
      const out = [];
      for (let i = 0; i < input.length; i++) {
        const a = input[i];
        const b = input[(i + 1) % input.length];
        const av = axis === 'x' ? a[0] : a[1];
        const bv = axis === 'x' ? b[0] : b[1];
        const ain = keepGreater ? av >= value - 1e-9 : av <= value + 1e-9;
        const bin = keepGreater ? bv >= value - 1e-9 : bv <= value + 1e-9;
        if (ain && bin) {
          out.push(b);
        } else if (ain !== bin) {
          const denom = bv - av;
          if (Math.abs(denom) > 1e-12) {
            const t = (value - av) / denom;
            out.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]);
          }
          if (!ain && bin) out.push(b);
        }
      }
      return out;
    }
    let out = poly.slice();
    out = clip(out, 'x', true, minX);
    out = clip(out, 'x', false, maxX);
    out = clip(out, 'y', true, minY);
    out = clip(out, 'y', false, maxY);
    return out.length >= 3 ? out : null;
  }
  function standDuplicateSafetyWorldPolygonForSpec(cx, cy, angleRad, depM, widM, category) {
    const polyLocal = buildStandDuplicateSafetyPolygonLocalPoints(depM, widM, category);
    if (!polyLocal || polyLocal.length < 3) return null;
    return polyLocal.map(function(p) { return standFootprintLocalToWorld(cx, cy, angleRad, p[0], p[1]); });
  }
  function standSafetyOverlapSpec(stand) {
    if (!stand || stand.id == null) return null;
    const id = String(stand.id);
    const isPbb = (state.pbbStands || []).some(function(s) { return s && String(s.id) === id; });
    const center = isPbb ? getStandConnectionPx(stand) : getRemoteStandCenterPx(stand);
    const angle = isPbb ? getPBBStandAngle(stand) : getRemoteStandAngleRad(stand);
    const category = stand.category || 'C';
    const dep = getStandDepthMeters(category);
    const wid = getStandWidthMeters(category);
    if (!center || !isFinite(center[0]) || !isFinite(center[1]) || !isFinite(dep) || !isFinite(wid)) return null;
    const poly = standDuplicateSafetyWorldPolygonForSpec(center[0], center[1], angle, dep, wid, category);
    const aabb = polygonAabbXY(poly);
    return poly && aabb ? { id: id, stand: stand, poly: poly, aabb: aabb } : null;
  }
  function recomputeDuplicateApronByStandId() {
    const specs = (typeof allStandsForFlightAssignment === 'function' ? allStandsForFlightAssignment() : [])
      .map(standSafetyOverlapSpec)
      .filter(Boolean);
    const map = {};
    for (let i = 0; i < specs.length; i++) {
      const a = specs[i];
      if (!map[a.id]) map[a.id] = [];
      for (let j = i + 1; j < specs.length; j++) {
        const b = specs[j];
        if (!aabbRectOverlap(a.aabb, b.aabb)) continue;
        if (!polygonsOverlapXY(a.poly, b.poly)) continue;
        if (!map[a.id]) map[a.id] = [];
        if (!map[b.id]) map[b.id] = [];
        map[a.id].push(b.id);
        map[b.id].push(a.id);
      }
    }
    specs.forEach(function(spec) {
      const list = (map[spec.id] || []).slice().sort();
      spec.stand.duplicate_apron_list = list;
      map[spec.id] = list;
    });
    state.duplicateApronByStandId = map;
    return map;
  }
  function duplicateApronStandIdsForStand(standId) {
    const key = standId != null ? String(standId) : '';
    if (!key) return [];
    const map = state.duplicateApronByStandId || {};
    const arr = Array.isArray(map[key]) ? map[key] : [];
    return arr.map(function(id) { return String(id); }).filter(Boolean);
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
