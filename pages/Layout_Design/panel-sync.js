    'DEP→DEP': { driver: 'Wake of leading departure aircraft', refPoint: 'Take-off / runway entry point of the leading departure', input: 'Lead (departure) × Trail (departure) matrix input' },
    'ARR→DEP': { driver: 'Leading aircraft ROT (runway occupancy time)', refPoint: 'Trailing aircraft: time from lineup to gear-off (lineup–gear-off)', input: 'Lead arrival category — 1D separation inputs' },
    'DEP→ARR': { driver: 'Wake / ROT of leading departure', refPoint: 'Runway vacation / ROT end of the leading departure', input: 'Trail (arrival category) 1‑D input' },
  };
  function rsepGetCatLabel(stdKey, cat) {
    const t = RSEP_CAT_LABELS[stdKey];
    if (!t) return '';
    return t[cat] || '';
  }
  function rsepGetSeqMeta(seq) {
    return RSEP_SEQ_META[seq] || null;
  }
  function _rsepStringValue(value) {
    return value != null ? String(value) : '';
  }
  function _rsepMakeCategoryValues(cats, src, asMatrix) {
    const out = {};
    cats.forEach(leadCat => {
      if (!asMatrix) {
        out[leadCat] = _rsepStringValue(src && src[leadCat]);
        return;
      }
      out[leadCat] = {};
      cats.forEach(trailCat => {
        out[leadCat][trailCat] = _rsepStringValue(src && src[leadCat] && src[leadCat][trailCat]);
      });
    });
    return out;
  }
  function rsepMakeMatrix(cats, src) {
    return _rsepMakeCategoryValues(cats, src, true);
  }
  function rsepMake1D(cats, src) {
    return _rsepMakeCategoryValues(cats, src, false);
  }
  function rsepMakeSeqData(stdKey) {
    const cats = RSEP_STD_CATS[stdKey] || [];
    const def = RSEP_DEFAULTS[stdKey] || {};
    const arrDep = rsepMake1D(cats, def['ARR→DEP']);
    const boost = RSEP_ARRDEP_BOOST_SEC;
    cats.forEach(function(c) {
      const s = arrDep[c];
      if (s === '' || s == null) return;
      const n = Number(s);
      if (isFinite(n)) arrDep[c] = String(Math.round(n + boost));
    });
    return {
      'ARR→ARR': rsepMakeMatrix(cats, def['ARR→ARR']),
      'DEP→DEP': rsepMakeMatrix(cats, def['DEP→DEP']),
      'ARR→DEP': arrDep,
      'DEP→ARR': rsepMake1D(cats, def['DEP→ARR']),
    };
  }

  function rsepColorForValue(val) {
    const n = Number(val);
    if (!isFinite(n) || val === '' || val == null) {
      return { bg: '#1a1a1a', color: '#e5e7eb', border: '#444444' };
    }
    const th = RSEP_COLOR_THRESHOLDS;
    for (let i = 0; i < th.length; i++) {
      if (n < th[i]) return RSEP_COLOR_STYLES[i] || RSEP_COLOR_STYLES[RSEP_COLOR_STYLES.length - 1];
    }
    return RSEP_COLOR_STYLES[th.length] || RSEP_COLOR_STYLES[RSEP_COLOR_STYLES.length - 1];
  }
  function rsepLegendHtml(filled, total) {
    const th = RSEP_COLOR_THRESHOLDS;
    const countColor = filled === total ? '#68d391' : '#9ca3af';
    let html = '<div style="display:flex;align-items:center;gap:12px;margin-top:4px;margin-bottom:4px;font-size:10px;color:#9ca3af;">';
    const lab = RSEP_LEGEND_LAB;
    if (th.length) {
      const st0 = rsepColorForValue(Math.max(0, th[0] - 1));
      html += '<span><span style="display:inline-block;width:10px;height:10px;background:' + st0.bg + ';border-radius:2px;margin-right:4px;"></span><span style="color:' + st0.color + ';">' + escapeHtml(rsepLegendFmt(lab.ltFirst || '<{0}s', th[0])) + '</span></span>';
      for (let i = 1; i < th.length; i++) {
        const lo = th[i - 1], hi = th[i];
        const mid = lo + (hi - lo) / 2;
        const st = rsepColorForValue(mid);
        const text = rsepLegendFmt(lab.rangeMid || '{0}–{1}s', lo, hi - 1);
        html += '<span><span style="display:inline-block;width:10px;height:10px;background:' + st.bg + ';border-radius:2px;margin-right:4px;"></span><span style="color:' + st.color + ';">' + escapeHtml(text) + '</span></span>';
      }
      const lastT = th[th.length - 1];
      const stL = rsepColorForValue(lastT + 1000);
      html += '<span><span style="display:inline-block;width:10px;height:10px;background:' + stL.bg + ';border-radius:2px;margin-right:4px;"></span><span style="color:' + stL.color + ';">' + escapeHtml(rsepLegendFmt(lab.gteLast || '≥{0}s', lastT)) + '</span></span>';
    }
    html += '<span style="margin-left:4px;color:' + countColor + ';">' + filled + '/' + total + '</span>';
    html += '</div>';
    return html;
  }
  function rsepMakeConfig(stdKey) {
    const std = RSEP_STANDARDS[stdKey] || RSEP_STANDARDS['ICAO'];
    const cats = RSEP_STD_CATS[stdKey];
    const rot = std.ROT || {};
    const rotCopy = {};
    const boost = RSEP_ARRDEP_BOOST_SEC;
    cats.forEach(function(c) {
      if (rot[c] == null || rot[c] === '') rotCopy[c] = '';
      else {
        const n = Number(rot[c]);


        rotCopy[c] = isFinite(n) ? String(Math.round(n + boost)) : String(rot[c]);
      }
    });
    return {
      standard: stdKey,
      mode: 'MIX',
      activeSeq: 'ARR→ARR',
      seqData: rsepMakeSeqData(stdKey),
      rot: rotCopy,
    };
  }
  function rsepGetConfigForRunway(rw) {
    if (!rw) return null;
    if (!rw.rwySepConfig) {
      rw.rwySepConfig = rsepMakeConfig('ICAO');
    }
    const cfg = rw.rwySepConfig;
    if (!RSEP_STD_CATS[cfg.standard]) {
      rw.rwySepConfig = rsepMakeConfig('ICAO');
      return rw.rwySepConfig;
    }
    return cfg;
  }
  let dpr = window.devicePixelRatio || 1;
  let ctx = (canvas && typeof canvas.getContext === 'function') ? canvas.getContext('2d') : null;
  let layoutDrawCanvas = canvas;

  function screenToWorld(sx, sy) {
    return [(sx - state.panX) / state.scale, (sy - state.panY) / state.scale];
  }
  function worldToScreenCanvas(wx, wy) {
    return [wx * state.scale + state.panX, wy * state.scale + state.panY];
  }
  function cellToPixel(col, row) { return [col * CELL_SIZE, row * CELL_SIZE]; }
  function getTaxiwayAvgMoveVelocityForPath(path) {
    if (path && typeof path.avgMoveVelocity === 'number' && isFinite(path.avgMoveVelocity) && path.avgMoveVelocity > 0)
      return Math.max(1, Math.min(50, path.avgMoveVelocity));
    const el = document.getElementById('taxiwayAvgMoveVelocity');
    const v = el ? Number(el.value) : 10;
    return (typeof v === 'number' && isFinite(v) && v > 0) ? Math.max(1, Math.min(50, v)) : 10;
  }
  function roundToStep(value, step) {
    const n = Number(value);
    const s = Number(step);
    if (!isFinite(n)) return 0;
    if (!isFinite(s) || s <= 0) return n;
    return Math.round(n / s) * s;
  }
  function clampToGridBounds(col, row) {
    const c = Math.max(0, Math.min(GRID_COLS, Number(col) || 0));
    const r = Math.max(0, Math.min(GRID_ROWS, Number(row) || 0));
    return [c, r];
  }
  function pixelToCell(x, y) {
    const cs = (typeof CELL_SIZE === 'number' && CELL_SIZE > 0) ? CELL_SIZE : 20;
    const snappedCol = roundToStep(x / cs, GRID_SNAP_STEP_CELL);
    const snappedRow = roundToStep(y / cs, GRID_SNAP_STEP_CELL);
    return clampToGridBounds(snappedCol, snappedRow);
  }
  function worldPointToCellPoint(wx, wy, snapToGrid) {
    const cs = (typeof CELL_SIZE === 'number' && CELL_SIZE > 0) ? CELL_SIZE : 20;
    const step = snapToGrid ? GRID_SNAP_STEP_CELL : FREE_DRAW_STEP_CELL;
    const col = roundToStep(wx / cs, step);
    const row = roundToStep(wy / cs, step);
    const clamped = clampToGridBounds(col, row);
    return { col: clamped[0], row: clamped[1] };
  }
  function worldPointToPixel(wx, wy, snapToGrid) {
    const pt = worldPointToCellPoint(wx, wy, snapToGrid);
    return cellToPixel(pt.col, pt.row);
  }
  const ICAO_STAND_SIZE_M = (function() {
    const m = _layoutTier.standSizesMByIcaoCategory;
    if (m && typeof m === 'object') {
      const o = {};
      Object.keys(m).forEach(k => { o[k] = Number(m[k]); });
      return o;
    }
    return { A: 20, B: 30, C: 40, D: 50, E: 60, F: 80 };
  })();
  function getStandSizeMeters(cat) { return ICAO_STAND_SIZE_M[cat] || 40; }
  const STAND_CONFIG_ROW_BY_CODE = (function() {
    const raw = _layoutTier.standConfig;
    const out = {};
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
    ctx.beginPath();
    ctx.moveTo(xNose, -noseHalf);
    ctx.lineTo(xNose, noseHalf);
    ctx.lineTo(xStop, noseHalf);
    ctx.lineTo(Math.min(xBendEnd, halfD), halfW);
    if (xBendEnd < halfD - eps) {
      ctx.lineTo(halfD, halfW);
      ctx.lineTo(halfD, -halfW);
      ctx.lineTo(xBendEnd, -halfW);
    } else {
      ctx.lineTo(halfD, -halfW);
    }
    ctx.lineTo(xStop, -noseHalf);
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
    ctx.strokeStyle = 'rgba(220,38,38,0.92)';
    ctx.lineWidth = Math.max(0.35, 0.42 / Math.max(state.scale, 0.1));
    ctx.setLineDash([2, 2.5]);
    ctx.lineCap = 'butt';
    ctx.lineJoin = 'miter';
    const nc = Number(r.nose_clear), pb = Number(r.pushback);
    if (isFinite(nc) && isFinite(pb)) {
      const xStop = -halfD + nc;
      const xPush = halfD - pb;
      if (xStop > -halfD + eps && xStop < halfD - eps) {
        ctx.beginPath();
        ctx.moveTo(xStop, -halfW);
        ctx.lineTo(xStop, halfW);
        ctx.stroke();
      }
      if (xPush < halfD - eps && xPush > -halfD + eps) {
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
        ctx.moveTo(-halfD, yLim);
        ctx.lineTo(halfD, yLim);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(-halfD, -yLim);
        ctx.lineTo(halfD, -yLim);
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
    ctx.beginPath();
    ctx.rect(-depM / 2, -widM / 2, depM, widM);
    ctx.fill();
  }
  function drawStandSafetyContourInLocalAxes(ctx, depM, widM, category, selected) {
    if (!buildStandSafetyPolygonPath(ctx, depM, widM, category)) return;
    ctx.save();
    ctx.strokeStyle = 'rgba(220,38,38,0.92)';
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
    const cat = (st && st.category) || 'C';
    const dep = getStandDepthMeters(cat);
    const wid = getStandWidthMeters(cat);
    const lx = standSafetyAircraftCenterLocalXM(dep, wid, cat);
    const cxy = getRemoteStandCenterPx(st);
    return standFootprintLocalToWorld(cxy[0], cxy[1], getRemoteStandAngleRad(st), lx, 0);
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
    const halfD = getStandDepthMeters(cat) / 2;
    const halfW = getStandWidthMeters(cat) / 2;
    const angle = getRemoteStandAngleRad(stLike);
    const cos = Math.cos(angle), sin = Math.sin(angle);
    return [
      [cx + (-halfD)*cos - (-halfW)*sin, cy + (-halfD)*sin + (-halfW)*cos],
      [cx + ( halfD)*cos - (-halfW)*sin, cy + ( halfD)*sin + (-halfW)*cos],
      [cx + ( halfD)*cos - ( halfW)*sin, cy + ( halfD)*sin + ( halfW)*cos],
      [cx + (-halfD)*cos - ( halfW)*sin, cy + (-halfD)*sin + ( halfW)*cos]
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
    const halfD = getStandDepthMeters(cat) / 2;
    const halfW = getStandWidthMeters(cat) / 2;
    const angle = getPBBStandAngle(pbb);
    const cos = Math.cos(angle), sin = Math.sin(angle);
    return [
      [cx + (-halfD)*cos - (-halfW)*sin, cy + (-halfD)*sin + (-halfW)*cos],
      [cx + ( halfD)*cos - (-halfW)*sin, cy + ( halfD)*sin + (-halfW)*cos],
      [cx + ( halfD)*cos - ( halfW)*sin, cy + ( halfD)*sin + ( halfW)*cos],
      [cx + (-halfD)*cos - ( halfW)*sin, cy + (-halfD)*sin + ( halfW)*cos]
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
