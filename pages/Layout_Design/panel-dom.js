    }
    let pose = getFlightPoseAtTime(flight, t);
    if (!pose) return null;
    pose = getPushbackReversePoseForDraw(flight, t, pose);
    pose = applyParkedStandHeadingToPoseIfNeeded(flight, t, pose);
    return pose;
  }
  function simFlightSilhouetteWorldPolygon(f, pose, tSecOpt) {
    if (!f || !pose) return [];
    const x = Number(pose.x), y = Number(pose.y), dx = Number(pose.dx), dy = Number(pose.dy);
    if (![x, y, dx, dy].every(isFinite)) return [];
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
    let scaleX, scaleY;
    if (useDetailSil && silhouette2D.length >= 3) {
      const sp = detailedSilhouetteAxisSpans(silhouette2D);
      scaleX = dimsM.lenM / sp.spanX;
      scaleY = dimsM.wingM / sp.spanY;
    } else {
      const xs = [nX, wRx, tX];
      const lenNorm = Math.max(1e-9, Math.max(xs[0], xs[1], xs[2]) - Math.min(xs[0], xs[1], xs[2]));
      const wingNorm = Math.max(1e-9, uY + lY);
      scaleX = dimsM.lenM / lenNorm;
      scaleY = dimsM.wingM / wingNorm;
    }
    let fuselageStationFrac = 0.15;
    if (typeof tSecOpt === 'number' && isFinite(tSecOpt) && simFlightPhaseAtTime(f, tSecOpt, pose) === 'Pushback') {
      fuselageStationFrac = 0.25;
    }
    const pFwX = nX * scaleX - fuselageStationFrac * dimsM.lenM;
    const drawX = x - nx * pFwX;
    const drawY = y - ny * pFwX;
    const pts = (useDetailSil && silhouette2D.length >= 3)
      ? silhouette2D.map(function(p) { return [p[0] * scaleX, p[1] * scaleY]; })
      : [[scaleX * nX, 0], [scaleX * wRx, scaleY * uY], [scaleX * tX, 0], [scaleX * wRx, scaleY * lY]];
    return pts.map(function(p) {
      return [drawX + p[0] * nx - p[1] * ny, drawY + p[0] * ny + p[1] * nx];
    });
  }
  function simFlightPhaseAtTime(f, tSec, pose) {
    if (pose && pose.phase != null) return String(pose.phase || '');
    const seg = typeof flightTimelineSegmentAtSimTime === 'function' ? flightTimelineSegmentAtSimTime(f, tSec) : null;
    if (!seg || !seg.a) return '';
    const pa = seg.a.phase != null ? String(seg.a.phase || '') : '';
    const pb = seg.b && seg.b.phase != null ? String(seg.b.phase || '') : pa;
    if (pa === 'Pushback' && pb === 'Pushback') return 'Pushback';
    if (pa === 'Pushback' && pb && pb !== 'Pushback') return pb;
    return pa || pb || '';
  }
  function isFlightParkedAtSimTime(f, tSec) {
    const m = f && f.timeline_meta;
    const t = Number(tSec);
    if (!m || !isFinite(t)) return false;
    const eibtList = Array.isArray(m.eibtSecList) ? m.eibtSecList : (typeof m.eibtSec === 'number' ? [m.eibtSec] : []);
    const eobtList = Array.isArray(m.eobtSecList) ? m.eobtSecList : (typeof m.eobtSec === 'number' ? [m.eobtSec] : []);
    const n = Math.min(eibtList.length, eobtList.length);
    for (let i = 0; i < n; i++) {
      const a = Number(eibtList[i]), b = Number(eobtList[i]);
      if (isFinite(a) && isFinite(b) && t >= a - 1e-3 && t <= b + 1e-3) return true;
    }
    return false;
  }
  function standIdForParkedApronInterval(f, tSec) {
    const m = f && f.timeline_meta;
    const t = Number(tSec);
    if (!m || !isFinite(t)) return null;
    const eibtList = Array.isArray(m.eibtSecList) ? m.eibtSecList : (typeof m.eibtSec === 'number' ? [m.eibtSec] : []);
    const eobtList = Array.isArray(m.eobtSecList) ? m.eobtSecList : (typeof m.eobtSec === 'number' ? [m.eobtSec] : []);
    const nInt = Math.min(eibtList.length, eobtList.length);
    let idx = -1;
    for (let i = 0; i < nInt; i++) {
      const a = Number(eibtList[i]), b = Number(eobtList[i]);
      if (isFinite(a) && isFinite(b) && t >= a - 1e-3 && t <= b + 1e-3) {
        idx = i;
        break;
      }
    }
    if (idx < 0) return null;
    const segs = Array.isArray(f.apronStaySegments) ? f.apronStaySegments : [];
    if (segs.length > idx && segs[idx] && segs[idx].standId != null && String(segs[idx].standId).trim() !== '') {
      return String(segs[idx].standId);
    }
    if (f.standId != null && String(f.standId).trim() !== '') return String(f.standId);
    return null;
  }
  /**
   * Formerly rotated nose to stand layout axis while on-block stationary; kept as a no-op so heading
   * comes only from timeline/pose logic (no velocity- or dwell-based direction overrides).
   */
  function applyParkedStandHeadingToPoseIfNeeded(flight, tSec, pose) {
    return pose;
  }
  /** EIBT–EOBT on-block stationary: pose unchanged for the dwell; skip repeat getFlightPoseAtTime sampling. */
  function getParkedOnBlockStationaryPoseCacheCtx(flight, tSec) {
    const t = Number(tSec);
    if (!flight || !isFinite(t)) return null;
    if (!isFlightParkedAtSimTime(flight, t)) return null;
    if (typeof isFlightTimelineStationaryAtSimTime !== 'function' || !isFlightTimelineStationaryAtSimTime(flight, t)) return null;
    const m = flight.timeline_meta;
    if (!m) return null;
    const eibtList = Array.isArray(m.eibtSecList) ? m.eibtSecList : (typeof m.eibtSec === 'number' ? [m.eibtSec] : []);
    const eobtList = Array.isArray(m.eobtSecList) ? m.eobtSecList : (typeof m.eobtSec === 'number' ? [m.eobtSec] : []);
    const nInt = Math.min(eibtList.length, eobtList.length);
    for (let i = 0; i < nInt; i++) {
      const a = Number(eibtList[i]), b = Number(eobtList[i]);
      if (!(isFinite(a) && isFinite(b) && t >= a - 1e-3 && t <= b + 1e-3)) continue;
      const sid = typeof standIdForParkedApronInterval === 'function' ? standIdForParkedApronInterval(flight, t) : '';
      let trTag = '|0|||';
      const cpt = typeof compactPlaybackTrackForFlight === 'function' ? compactPlaybackTrackForFlight(flight) : null;
      if (cpt && Array.isArray(cpt.t) && cpt.t.length) {
        trTag = '|cp|' + cpt.t.length + '|' + cpt.t[0] + '|' + cpt.t[cpt.t.length - 1];
      } else if (flight.timeline && flight.timeline.length) {
        const tl0 = flight.timeline[0], tlZ = flight.timeline[flight.timeline.length - 1];
        trTag = '|tl|' + flight.timeline.length + '|' + tl0.t + '|' + tlZ.t;
      }
      const key = String(flight.id) + '|' + String(sid || '') + '|' + i + '|' + a + '|' + b + trTag;
      return { key: key, anchorT: a };
    }
    return null;
  }
  function isSecondOrLaterArrTaxiAtTime(f, tSec) {
    const tl = f && Array.isArray(f.timeline) ? f.timeline : null;
    const t = Number(tSec);
    if (!tl || !tl.length || !isFinite(t)) return true;
    let arrTaxiBlockCount = 0;
    let prevArrTaxi = false;
    for (let i = 0; i < tl.length; i++) {
      const ti = Number(tl[i].t);
      if (!isFinite(ti) || ti > t + 1e-9) break;
      const ph = String(tl[i].phase || '').toLowerCase();
      const isArrTaxi = ph.indexOf('arr_taxi') >= 0 || ph.indexOf('arr taxi') >= 0;
      if (isArrTaxi && !prevArrTaxi) arrTaxiBlockCount++;
      prevArrTaxi = isArrTaxi;
    }
    return arrTaxiBlockCount >= 2;
  }
  function flightNeedsTugAtSimTime(f, tSec, pose) {
    if (!f) return false;
    const tr = typeof compactPlaybackTrackForFlight === 'function' ? compactPlaybackTrackForFlight(f) : null;
    if (tr) return compactPlaybackNeedsTugAt(tr, tSec);
    return false;
  }
  function drawFlightTugCar2D(ctx, x, y, nx, ny, lenM, wingM) {
    void lenM;
    void wingM;
    const tugLen = 8;
    const tugWid = 3;
    const cx = x + nx * 3.4;
    const cy = y + ny * 3.4;
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(Math.atan2(ny, nx));
    ctx.fillStyle = '#22c55e';
    ctx.strokeStyle = 'rgba(5,46,22,0.95)';
    ctx.lineWidth = Math.max(0.6, 0.9 / Math.max(state.scale, 0.1));
    ctx.beginPath();
    ctx.rect(-tugLen / 2, -tugWid / 2, tugLen, tugWid);
    ctx.fill();
    ctx.stroke();
    ctx.restore();
  }

  function isFlightPreTouchdownForDraw(f, tSec) {
    if (!PRE_TOUCHDOWN_HALO_ENABLED) return false;
    if (!f || f.arrDep === 'Dep') return false;
    const m = f.timeline_meta;
    if (!m || typeof m.eldtSec !== 'number' || !isFinite(m.eldtSec)) return false;
    const t = Number(tSec);
    if (!isFinite(t)) return false;
    return t < m.eldtSec - 1e-3;
  }

  function isFlightAirsideCycleCompleteAtSimTime(f, tSec) {
    const m = f && f.timeline_meta;
    const t = Number(tSec);
    if (!isFinite(t) || !m || m.error) return false;
    if (typeof m.etotSec !== 'number' || !isFinite(m.etotSec)) return false;
    return t >= m.etotSec - 1e-3;
  }

  
  function isFlightTimelineStationaryAtSimTime(f, tSec) {
    const tr = compactPlaybackTrackForFlight(f);
    const tl = tr ? compactPlaybackTimelineWindow(tr, tSec, 2) : (f && f.timeline);
    if (!tl || tl.length < 2) return false;
    const t = Number(tSec);
    if (!isFinite(t)) return false;
    const t0 = tl[0].t, t1 = tl[tl.length - 1].t;
    if (t < t0 - 1e-9 || t > t1 + 1e-9) return false;
    const stillEps = 0.08;
    for (let i = 0; i < tl.length - 1; i++) {
      const a = tl[i], b = tl[i + 1];
      if (!(t + 1e-9 >= a.t && t - 1e-9 <= b.t)) continue;
      const dt = b.t - a.t;
      if (dt < 1e-9) continue;
      const dist = Math.hypot(b.x - a.x, b.y - a.y);
      if (dist < stillEps) return true;
    }
    return false;
  }

  function isFlightTrailHiddenAtSimTime(f, tSec) {
    if (isFlightAirsideCycleCompleteAtSimTime(f, tSec)) return true;
    if (isFlightTimelineStationaryAtSimTime(f, tSec)) return true;
    return false;
  }

  function getFlightTrailPolylineBackward(f, tEnd, maxDistM) {
    const tr = compactPlaybackTrackForFlight(f);
    const tl = tr ? compactPlaybackTimelineWindow(tr, tEnd, 160) : (f && f.timeline);
    if (!tl || tl.length < 2 || !(maxDistM > 0)) return [];
    const tMin = tl[0].t, tMax = tl[tl.length - 1].t;
    let t = Math.min(Math.max(tEnd, tMin), tMax);
    let seg = Math.max(0, timelineSegmentIndexAtTime(tl, t, true));
    const pts = [];
    function xyAt(T) {
      if (T <= tMin) return [tl[0].x, tl[0].y];
      if (T >= tMax) return [tl[tl.length - 1].x, tl[tl.length - 1].y];
      const i = timelineSegmentIndexAtTime(tl, T, true);
      if (i >= 0) {
        const a = tl[i], b = tl[i + 1];
        const sp = b.t - a.t || 1;
        const uu = (T - a.t) / sp;
        return [a.x + (b.x - a.x) * uu, a.y + (b.y - a.y) * uu];
      }
      return [tl[tl.length - 1].x, tl[tl.length - 1].y];
    }
    pts.push(xyAt(t));
    let rem = maxDistM;
    let curSeg = seg;
    let curT = t;
    let guard = 0;
    while (rem > 1e-6 && curSeg >= 0 && guard++ < 10000) {
      const A = tl[curSeg], B = tl[curSeg + 1];
      const ta = A.t, tb = B.t;
      const dt = tb - ta || 1e-12;
      const distAB = Math.hypot(B.x - A.x, B.y - A.y) || 1e-12;
      let u = Math.max(0, Math.min(1, (curT - ta) / dt));
      if (u < 1e-12) {
        if (curSeg <= 0) break;
        curSeg--;
        curT = tl[curSeg + 1].t;
        continue;
      }
      const distToA = u * distAB;
      if (distToA <= rem) {
        rem -= distToA;
        pts.push([A.x, A.y]);
        curSeg--;
        curT = ta;
      } else {
        const frac = rem / distAB;
        const uu = u - frac;
        const nx = A.x + uu * (B.x - A.x);
        const ny = A.y + uu * (B.y - A.y);
        pts.push([nx, ny]);
        rem = 0;
        break;
      }
    }
    return pts.slice().reverse();
  }

  function getRunwayOptions() {
    const list = [];
    (state.taxiways || []).filter(t => t.pathType === 'runway')
      .forEach(t => list.push({ id: t.id, name: (t.name || '').trim() || 'Runway' }));
    return list;
  }

  function buildRunwayOptionsHtml(selectedId) {
    const opts = [];
    const list = getRunwayOptions();
    if (!list.length) {
      opts.push('<option value=\"\">Runway</option>');
    } else {
      list.forEach(function(o) {
        const sel = selectedId && o.id === selectedId ? ' selected' : '';
        opts.push('<option value=\"' + String(o.id || '').replace(/\"/g, '&quot;') + '\"' + sel + '>' +
          escapeHtml(o.name || o.id || 'Runway') + '</option>');
      });
    }
    return opts.join('');
  }
  function buildTerminalOptionsHtml(selectedId) {
    const opts = [];
    const terms = makeUniqueNamedCopy(state.terminals || [], 'name').map(function(t) {
      return { id: t.id, name: (t.name || '').trim() || 'Building' };
    });
    if (!terms.length) {
      opts.push('<option value=\"\">Building</option>');
    } else {
      if (terms.length > 1) opts.push('<option value=\"\">Random</option>');
      terms.forEach(function(o) {
        const sel = selectedId && o.id === selectedId ? ' selected' : '';
        opts.push('<option value=\"' + String(o.id || '').replace(/\"/g, '&quot;') + '\"' + sel + '>' +
          escapeHtml(o.name || o.id || 'Building') + '</option>');
      });
    }
    return opts.join('');
  }
  function resolveRunwayIdFromInput(raw) {
    const v = (raw || '').trim();
    if (!v) return null;
    const list = getRunwayOptions();
    for (let i = 0; i < list.length; i++) {
      if (list[i].id === v) return v;
    }
    const vl = v.toLowerCase();
    for (let i = 0; i < list.length; i++) {
      if (String(list[i].name || '').trim().toLowerCase() === vl) return list[i].id;
    }
    return undefined;
  }
  function resolveTerminalIdFromInput(raw) {
    const v = (raw || '').trim();
    if (!v) return null;
    const terms = makeUniqueNamedCopy(state.terminals || [], 'name');
    for (let i = 0; i < terms.length; i++) {
      const t = terms[i];
      if (t.id === v) return v;
    }
    const vl = v.toLowerCase();
    for (let i = 0; i < terms.length; i++) {
      const t = terms[i];
      if (String(t.name || '').trim().toLowerCase() === vl) return t.id;
    }
    return undefined;
  }
  function syncFlightAssignInputDisplay(el, f) {
    const role = el.getAttribute('data-role');
    if (role === 'arr') el.value = resolveArrivalRunwayIdForFlight(f) || '';
    else if (role === 'term' || role === 'arrterm') el.value = resolveFlightArrTerminalId(f) || '';
    else if (role === 'depterm') el.value = resolveFlightDepTerminalId(f) || '';
    else if (role === 'dep') el.value = f.depRunwayId || (f.token && f.token.depRunwayId) || '';
    else if (role === 'intdom') el.value = (f && String(f.intDom || '').trim().toLowerCase() === 'dom') ? 'Dom' : 'Int';
  }
  function getRunwayDisplayLabelById(rwId) {
    if (rwId == null || rwId === '') return '—';
    const list = getRunwayOptions();
    const o = list.find(function(x) { return x.id === rwId; });
    return o ? (o.name || o.id || 'Runway') : '—';
  }
  function getTerminalDisplayLabelById(termId) {
    if (termId == null || termId === '') return '—';
    const terms = makeUniqueNamedCopy(state.terminals || [], 'name');
    const t = terms.find(function(x) { return x.id === termId; });
    return t ? ((t.name || '').trim() || 'Building') : '—';
  }
  function resolveFlightBaseTerminalId(f) {
    if (!f) return null;
    return f.terminalId || (f.token && f.token.terminalId) || null;
  }
  function resolveFlightArrTerminalId(f) {
    if (!f) return null;
    return f.arrTerminalId || (f.token && f.token.arrTerminalId) || resolveFlightBaseTerminalId(f);
  }
  function resolveFlightDepTerminalId(f) {
    if (!f) return null;
    return f.depTerminalId || (f.token && f.token.depTerminalId) || resolveFlightBaseTerminalId(f);
  }
  function ensureFlightSplitTerminalDefaults(f) {
    if (!f) return;
    const base = resolveFlightBaseTerminalId(f);
    if (!f.arrTerminalId && base) f.arrTerminalId = base;
    if (!f.depTerminalId && base) f.depTerminalId = base;
    if (f.token) {
      if (!f.token.arrTerminalId && f.arrTerminalId) f.token.arrTerminalId = f.arrTerminalId;
      if (!f.token.depTerminalId && f.depTerminalId) f.token.depTerminalId = f.depTerminalId;
    }
  }
  function flightColorGroupKeyForSim(f, mode) {
    if (mode === 'all') return '*';
    if (mode === 'airline') return 'a:' + (String(f.airlineCode || '').trim() || '—');
    if (mode === 'icao') {
      const c0 = (typeof getCodeForAircraft === 'function') ? String(getCodeForAircraft(f.aircraftType) || 'C').trim().toUpperCase()[0] : 'C';
      return 'i:' + (c0 || 'C');
    }
    if (mode === 'intdom') {
      return 'd:' + ((String(f.intDom || '').trim().toLowerCase() === 'dom') ? 'Dom' : 'Int');
    }
    if (mode === 'building') {
      const arrTid = resolveFlightArrTerminalId(f) || '';
      const depTid = resolveFlightDepTerminalId(f) || '';
      const arrLab = arrTid ? getTerminalDisplayLabelById(arrTid) : '—';
      const depLab = depTid ? getTerminalDisplayLabelById(depTid) : arrLab;
      return 'b:' + arrLab + ' / ' + depLab;
    }
    return '*';
  }
  function buildFlightSim2DColorKeyIndexMap() {
    const mode = state.flightColorMode || 'all';
    if (mode === 'all') return new Map([['*', 0]]);
    const flights = state.flights || [];
    const keys = new Set();
    for (let i = 0; i < flights.length; i++) {
      if (!flights[i]) continue;
      keys.add(flightColorGroupKeyForSim(flights[i], mode));
    }
    const sorted = Array.from(keys).sort();
    const m = new Map();
    for (let j = 0; j < sorted.length; j++) m.set(sorted[j], j);
    return m;
  }
  function resolveFlightSim2DGlyphFillRgba(f, isDeadlockGhost, keyIdxMap, pal, overflow, mode) {
    if (isDeadlockGhost) return 'rgba(148, 163, 184, 0.45)';
    if (mode === 'all') return apron2DGlyphFill();
    const k = flightColorGroupKeyForSim(f, mode);
    const idx = keyIdxMap.get(k);
    if (idx == null || idx >= 10) return overflow;
    return pal[idx] || overflow;
  }
  function parseCssColorToRgbOptional(css) {
    const s = String(css || '').trim();
    const hex6 = s.match(/^#([0-9a-fA-F]{6})$/);
    if (hex6) {
      const h = hex6[1];
      return { r: parseInt(h.slice(0, 2), 16), g: parseInt(h.slice(2, 4), 16), b: parseInt(h.slice(4, 6), 16) };
    }
    const rgba = s.match(/^rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)/);
    if (rgba) return { r: +rgba[1], g: +rgba[2], b: +rgba[3] };
    return null;
  }
  /** Trail stroke gradient: same hue as aircraft fill, fading to transparent along the tail. */
  function simFlightTrailGradientFromFillCss(fillCss) {
    const rgb = parseCssColorToRgbOptional(fillCss);
    if (!rgb) {
      return { near: c2dSimFlightTrailStroke(), far: c2dSimFlightTrailStrokeEnd() };
    }
    return {
      near: 'rgba(' + rgb.r + ',' + rgb.g + ',' + rgb.b + ',0.96)',
      far: 'rgba(' + rgb.r + ',' + rgb.g + ',' + rgb.b + ',0)',
    };
  }
  /** Pre-TD ring: same hue as fill, with soft fill + stroke + glow. */
  function simPreTouchdownHaloFromFillCss(fillCss) {
    const rgb = parseCssColorToRgbOptional(fillCss);
    if (!rgb) {
      return {
        fill: c2dSimPreTouchdownHaloFill(),
        stroke: c2dSimPreTouchdownHaloStroke(),
        shadow: c2dSimPreTouchdownHaloStroke(),
      };
    }
    return {
      fill: 'rgba(' + rgb.r + ',' + rgb.g + ',' + rgb.b + ',0.18)',
      stroke: 'rgba(' + rgb.r + ',' + rgb.g + ',' + rgb.b + ',0.92)',
      shadow: 'rgba(' + rgb.r + ',' + rgb.g + ',' + rgb.b + ',0.55)',
    };
  }
  function syncFlightAssignStripFromFlight(f) {
    const arrEl = document.getElementById('flightAssignStripArr');
    const arrTermEl = document.getElementById('flightAssignStripArrTerm');
    const depTermEl = document.getElementById('flightAssignStripDepTerm');
    const depEl = document.getElementById('flightAssignStripDep');
    const intDomEl = document.getElementById('flightAssignStripIntDom');
    const laArrInp = document.getElementById('flightLookaheadArrInput');
    const laDepInp = document.getElementById('flightLookaheadDepInput');
    if (f) ensureFlightSplitTerminalDefaults(f);
    if (arrEl) {
      const sid = f ? (resolveArrivalRunwayIdForFlight(f) || '') : '';
      arrEl.innerHTML = buildRunwayOptionsHtml(sid);
      arrEl.value = sid;
    }
    if (intDomEl) {
      intDomEl.value = (f && String(f.intDom || '').trim().toLowerCase() === 'dom') ? 'Dom' : 'Int';
    }
    if (arrTermEl) {
      const tid = f ? (resolveFlightArrTerminalId(f) || '') : '';
      arrTermEl.innerHTML = buildTerminalOptionsHtml(tid);
      arrTermEl.value = tid;
    }
    if (depTermEl) {
      const tid = f ? (resolveFlightDepTerminalId(f) || '') : '';
      depTermEl.innerHTML = buildTerminalOptionsHtml(tid);
      depTermEl.value = tid;
    }
    if (depEl) {
      const did = f ? (f.depRunwayId || (f.token && f.token.depRunwayId) || '') : '';
      depEl.innerHTML = buildRunwayOptionsHtml(did);
      depEl.value = did;
    }
    if (f) ensureFlightLookaheadArrDepFlight(f);
    if (laArrInp) {
      if (!f) {
        laArrInp.value = '9';
      } else {
        let va = f.lookaheadArr;
        if (va == null || va === '' || !isFinite(Number(va))) va = 9;
        else va = Math.max(0, Math.min(200, Math.floor(Number(va))));
        laArrInp.value = String(va);
      }
    }
    if (laDepInp) {
      if (!f) {
        laDepInp.value = '9';
      } else {
        let vd = f.lookaheadDep;
        if (vd == null || vd === '' || !isFinite(Number(vd))) vd = 9;
        else vd = Math.max(0, Math.min(200, Math.floor(Number(vd))));
        laDepInp.value = String(vd);
      }
    }
  }
  function syncFlightAssignStrip() {
    const arrEl = document.getElementById('flightAssignStripArr');
    const arrTermEl = document.getElementById('flightAssignStripArrTerm');
    const depTermEl = document.getElementById('flightAssignStripDepTerm');
    const depEl = document.getElementById('flightAssignStripDep');
    const intDomEl = document.getElementById('flightAssignStripIntDom');
    const laArrInp = document.getElementById('flightLookaheadArrInput');
    const laDepInp = document.getElementById('flightLookaheadDepInput');
    const sel = state.selectedObject;
    const hasFlight = sel && sel.type === 'flight' && sel.id;
    const f = hasFlight ? state.flights.find(function(x) { return x.id === sel.id; }) : null;
    const dis = !f;
    [arrEl, arrTermEl, depTermEl, depEl, intDomEl, laArrInp, laDepInp].forEach(function(el) {
      if (el) el.disabled = dis;
    });
    if (!f) {
      syncFlightAssignStripFromFlight(null);
      return;
    }
    syncFlightAssignStripFromFlight(f);
  }
  function commitFlightAssign(role, flightId, rawValue, st, listEl) {
    const f = st.flights.find(function(x) { return x.id === flightId; });
    if (!f) return;
    const raw = rawValue;
    if (role === 'intdom') {
      const next = (String(raw || '').trim().toLowerCase() === 'dom') ? 'Dom' : 'Int';
      const prev = (String(f.intDom || '').trim().toLowerCase() === 'dom') ? 'Dom' : 'Int';
      if (next === prev) return;
      f.intDom = next;
      syncFlightAssignStripFromFlight(f);
      if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
      if (typeof draw === 'function') draw();
      if (typeof renderFlightList === 'function')
        renderFlightList(false, false, { scheduleMode: 'incremental', dirtyFlightIds: [flightId], touchedStandIds: f.standId ? [f.standId] : [] });
      return;
    }
    var val = null;
    if (role === 'arr' || role === 'dep') {
      const r = resolveRunwayIdFromInput(raw);
      if ((raw || '').trim() && r === undefined) {
        syncFlightAssignStripFromFlight(f);
        return;
      }
      val = r === undefined ? null : r;
    } else if (role === 'term' || role === 'arrterm' || role === 'depterm') {
      const r = resolveTerminalIdFromInput(raw);
      if ((raw || '').trim() && r === undefined) {
        syncFlightAssignStripFromFlight(f);
        return;
      }
      val = r === undefined ? null : r;
    } else return;
    var prevArr = f.arrRunwayId || null;
    var prevDep = f.depRunwayId || (f.token && f.token.depRunwayId) || null;
    var prevArrTerm = resolveFlightArrTerminalId(f) || null;
    var prevDepTerm = resolveFlightDepTerminalId(f) || null;
    if (role === 'arr' && val === prevArr) return;
    if (role === 'dep' && val === prevDep) return;
    if ((role === 'term' || role === 'arrterm') && val === prevArrTerm) return;
    if (role === 'depterm' && val === prevDepTerm) return;
    var prevStand = f.standId || null;
    if (!f.token) f.token = { nodes: ['runway','taxiway','apron','terminal'], runwayId: null, apronId: null, terminalId: null, arrTerminalId: null, depTerminalId: null };
    if (role === 'arr') {
      f.arrRunwayId = val;
      f.token.runwayId = val;
    } else if (role === 'term' || role === 'arrterm') {
      f.arrTerminalId = val;
      f.token.arrTerminalId = val;
      if (!f.depTerminalId) {
        f.depTerminalId = val;
        f.token.depTerminalId = val;
      }
      f.terminalId = val;
      f.token.terminalId = val;
    } else if (role === 'depterm') {
      f.depTerminalId = val;
      f.token.depTerminalId = val;
      if (!f.arrTerminalId) {
        f.arrTerminalId = val;
        f.token.arrTerminalId = val;
      }
      f.terminalId = f.arrTerminalId || val;
      f.token.terminalId = f.terminalId || null;
    } else if (role === 'dep') {
      f.depRunwayId = val;
      f.token.depRunwayId = val;
    }
    syncFlightAssignStripFromFlight(f);
    if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
    var touched = [];
    if (prevStand) touched.push(prevStand);
    if (f.standId) touched.push(f.standId);
    if (typeof renderFlightList === 'function')
      renderFlightList(false, false, { scheduleMode: 'incremental', dirtyFlightIds: [flightId], touchedStandIds: touched });
  }
  function commitFlightAssignField(el, st, listEl) {
    const idVal = el.getAttribute('data-id');
    const role = el.getAttribute('data-role');
    commitFlightAssign(role, idVal, el.value, st, listEl);
  }
  function commitFlightAssignFromStrip(el, st, listEl) {
    const sel = state.selectedObject;
    if (!sel || sel.type !== 'flight' || !sel.id) return;
    const role = el.getAttribute('data-role');
    if (!role) return;
    commitFlightAssign(role, sel.id, el.value, st, listEl);
  }

  /** Flight schedule: 10 fixed, AP×k, Lookahead_arr, Lookahead_dep, Dep Rw, S/E blocks. */
  const FLIGHT_SCHED_FIXED_BEFORE_AP_COL_COUNT = 10;
  const FLIGHT_SCHED_TRAILING_METRIC_COL_COUNT = 7;
  function flightScheduleLogicalSegmentCount(f) {
    if (!f) return 1;
    const segs = typeof normalizeFlightApronStaySegments === 'function' ? normalizeFlightApronStaySegments(f) : [];
    if (!segs.length) return 1;
    let n = 0;
    let prev = null;
    for (let i = 0; i < segs.length; i++) {
      const sid = segs[i].standId != null ? String(segs[i].standId) : '';
      if (i === 0 || sid !== prev) n++;
      prev = sid;
    }
    return Math.max(1, n);
  }
  function flightScheduleColumnK() {
    const flights = state.flights || [];
    let k = 1;
    for (let i = 0; i < flights.length; i++) k = Math.max(k, flightScheduleLogicalSegmentCount(flights[i]));
    return k;
  }
  function flightSchedColIndex(field, k) {
    const n = Math.max(1, Number(k) || flightScheduleColumnK());
    const apStart = FLIGHT_SCHED_FIXED_BEFORE_AP_COL_COUNT;
    const base = apStart + n + 3;
    if (field === 'ap') return apStart;
    if (field === 'lookaheadArr') return apStart + n;
    if (field === 'lookaheadDep') return apStart + n + 1;
    if (field === 'depRunway') return apStart + n + 2;
    if (field === 'sibt') return base;
    if (field === 'sobt') return base + 1;
    if (field === 'eldt') return base + n * 2;
    if (field === 'eibt') return base + n * 2 + 1;
    if (field === 'eobt') return base + n * 2 + 2;
    if (field === 'etot') return base + n * 4 + 1;
    if (field === 'metrics') return base + n * 4 + 2;
    return base;
  }
  function flightScheduleTableColCount(k) {
    return flightSchedColIndex('metrics', k) + FLIGHT_SCHED_TRAILING_METRIC_COL_COUNT + 1;
  }
  function ensureFlightAssignStripWired() {
    if (window.__flightAssignStripWired) return;
    const wrap = document.getElementById('flightAssignStrip');
    if (!wrap) return;
    window.__flightAssignStripWired = true;
    wrap.querySelectorAll('.flight-assign-strip-select').forEach(function(inp) {
      inp.addEventListener('change', function(ev) {
        const listEl = document.getElementById('flightList');
        const el = ev.target;
        commitFlightAssignFromStrip(el, state, listEl);
      });
    });
    function wireLookaheadInput(el, setter) {
      if (!el || el._lookaheadArrDepWired) return;
      el._lookaheadArrDepWired = true;
      el.addEventListener('change', function() {
        if (!state.selectedObject || state.selectedObject.type !== 'flight') return;
        const ff = state.selectedObject.obj;
        let v = parseInt(String(this.value != null ? this.value : '9'), 10);
        if (!isFinite(v)) v = 9;
        v = Math.max(0, Math.min(200, v));
        setter(ff, v);
        this.value = String(v);
        ensureFlightLookaheadArrDepFlight(ff);
        if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
        if (typeof renderFlightList === 'function')
          renderFlightList(false, false, { scheduleMode: 'incremental', dirtyFlightIds: [ff.id], touchedStandIds: ff.standId ? [ff.standId] : [] });
      });
    }
    const laArr0 = document.getElementById('flightLookaheadArrInput');
    wireLookaheadInput(laArr0, function(ff, v) { ff.lookaheadArr = v; });
    const laDep0 = document.getElementById('flightLookaheadDepInput');
    wireLookaheadInput(laDep0, function(ff, v) { ff.lookaheadDep = v; });
  }

  function _flightListSortedFlightsCopy() {
    const flightsSorted = state.flights.slice();
    flightsSorted.sort(function(a, b) {
      return (a.sibtMin != null ? a.sibtMin : (a.timeMin != null ? a.timeMin : 0)) -
        (b.sibtMin != null ? b.sibtMin : (b.timeMin != null ? b.timeMin : 0));
    });
    return flightsSorted;
  }
  function _flightListSortedIndexForFlightId(flightsSorted, flightId) {
    const want = String(flightId);
    for (let i = 0; i < flightsSorted.length; i++) {
      const f = flightsSorted[i];
      if (f && String(f.id) === want) return i;
    }
    return -1;
  }
  /** Match Flight Schedule row highlight (purple) to ``state.selectedObject`` flight; optional scroll when ``scrollRow``. */
  function _flightListApplyScheduleSelectionHighlightDom(listEl, scrollRow) {
    if (!listEl) return;
    listEl.querySelectorAll('.flight-schedule-table tbody tr.obj-item').forEach(function(r) {
      r.classList.remove('selected', 'expanded');
    });
    const sel = state.selectedObject;
    if (!sel || sel.type !== 'flight' || sel.id == null) return;
    const row = listEl.querySelector('.flight-schedule-table tbody tr.obj-item[data-id="' + String(sel.id) + '"]');
    if (!row) return;
    row.classList.add('selected', 'expanded');
    if (scrollRow) {
      try {
        row.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
      } catch (eScroll) {
        row.scrollIntoView(false);
      }
    }
  }
  /** Grid / external selection: jump pager & virtual scroll so the flight row exists, then highlight. */
  function syncFlightScheduleTableSelectionHighlight() {
    const listEl = document.getElementById('flightList');
    if (!listEl) return;
    const sel = state.selectedObject;
    if (!sel || sel.type !== 'flight' || sel.id == null) {
      _flightListApplyScheduleSelectionHighlightDom(listEl, false);
      return;
    }
    const flightsSorted = _flightListSortedFlightsCopy();
    const idx = _flightListSortedIndexForFlightId(flightsSorted, sel.id);
    if (idx < 0) {
      _flightListApplyScheduleSelectionHighlightDom(listEl, false);
      return;
    }
    const size = FLIGHT_SCHED_PAGE_SIZE;
    const usePagination = size > 0;
    if (usePagination) {
      const targetPage = Math.floor(idx / size);
      if (state.flightSchedulePage !== targetPage) {
        state.flightSchedulePage = targetPage;
        if (typeof renderFlightList === 'function')
          renderFlightList(false, false, { pageTurnOnly: true });
        _flightListApplyScheduleSelectionHighlightDom(listEl, true);
        return;
      }
    }
    const vs = listEl._flightVirtState;
    if (vs && flightsSorted.length && !usePagination) {
      const rowH = vs.rowH || DOM_OPT_FLIGHT_VIRT_ROW_H;
      const vh = listEl.clientHeight || 418;
      listEl.scrollTop = Math.max(0, idx * rowH - Math.max(0, (vh - rowH) * 0.5));
      _flightListPaintVirtualSlice(listEl);
      _flightListApplyScheduleSelectionHighlightDom(listEl, true);
      return;
    }
    _flightListApplyScheduleSelectionHighlightDom(listEl, true);
  }

  function _flightListPaintVirtualSlice(listEl) {
    const vs = listEl._flightVirtState;
    if (!vs) return;
    const tbody = listEl.querySelector('.flight-schedule-table[data-virtual-table=\"1\"] tbody');
    if (!tbody) return;
    const flightsSorted = vs.flightsSorted;
    const retStatsAll = vs.retStatsAll;
    const total = flightsSorted.length;
    const rowH = vs.rowH;
    const overscan = vs.overscan;
    const scrollTop = listEl.scrollTop || 0;
    const vh = listEl.clientHeight || 418;
    const start = Math.max(0, Math.floor(scrollTop / rowH) - overscan);
    const rowCount = Math.ceil(vh / rowH) + overscan * 2 + 2;
    const end = Math.min(total, start + rowCount);
    const topPad = start * rowH;
    const botPad = Math.max(0, (total - end) * rowH);
    const parts = [];
    const colCount = flightScheduleTableColCount(vs.apronK || flightScheduleColumnK());
    parts.push('<tr class=\"flight-virt-spacer\" aria-hidden=\"true\" style=\"height:' + topPad + 'px\"><td colspan=\"' + colCount + '\"></td></tr>');
    for (let i = start; i < end; i++) {
      parts.push(_buildFlightListRowHtml(flightsSorted[i], retStatsAll, vs.apronK));
    }
    parts.push('<tr class=\"flight-virt-spacer\" aria-hidden=\"true\" style=\"height:' + botPad + 'px\"><td colspan=\"' + colCount + '\"></td></tr>');
    tbody.innerHTML = parts.join('');
    _flightListWireEvents(listEl, state);
    _flightListApplyScheduleSelectionHighlightDom(listEl, false);
  }
  function _flightListTeardownVirtual(listEl) {
    listEl._flightVirtState = null;
  }
  function _flightListMountVirtual(listEl, flightsSorted, retStatsAll, headerRow, apronK) {
    const prevScroll = listEl.querySelector('.flight-schedule-table[data-virtual-table=\"1\"]') ? (listEl.scrollTop || 0) : 0;
    listEl._flightVirtState = {
      flightsSorted: flightsSorted,
      retStatsAll: retStatsAll,
      rowH: DOM_OPT_FLIGHT_VIRT_ROW_H,
      overscan: DOM_OPT_FLIGHT_VIRT_OVERSCAN,
      apronK: apronK,
      raf: null
    };
    listEl.innerHTML = headerRow + '</tbody></table>';
    const tbl = listEl.querySelector('.flight-schedule-table');
    if (tbl) tbl.setAttribute('data-virtual-table', '1');
    _flightListPaintVirtualSlice(listEl);
    if (prevScroll > 0) listEl.scrollTop = prevScroll;
    if (!listEl._flightVirtScrollBound) {
      listEl._flightVirtScrollBound = true;
      listEl.addEventListener('scroll', function() {
        const vs = listEl._flightVirtState;
        if (!vs || !listEl.querySelector('.flight-schedule-table[data-virtual-table=\"1\"]')) return;
        if (vs.raf) cancelAnimationFrame(vs.raf);
        vs.raf = requestAnimationFrame(function() {
          vs.raf = null;
          _flightListPaintVirtualSlice(listEl);
        });
      });
    }
  }

  function bumpVttArrCacheRev() {
    state.vttArrCacheRev = (state.vttArrCacheRev | 0) + 1;
    bumpRwySepSnapshotStaleGen();
  }
  function getBaseVttArrMinutes(f) {
    if (!f) return 0;
    return 0;
  }
  function getArrRotMinutes(f) {
    if (!f) return 0;
    return 0;
  }
  function getBaseVttDepMinutes(f) {
    if (!f) return 0;
    return 0;
  }
  
  function getBaseVttDepMinutesToLineup(f) {
    if (!f) return 0;
    return 0;
  }
  
  function getDepBlockOutMin(f) {
    const taxi = (typeof getBaseVttDepMinutesToLineup === 'function') ? getBaseVttDepMinutesToLineup(f) : 0;
    const rollBundleSec = (typeof computeDepRollAndLineupOnlySec === 'function')
      ? computeDepRollAndLineupOnlySec(f)
      : (DEP_LINEUP_HOLD_SEC + takeoffRollSecForRunwayTailLenM(0, DEP_TAKEOFF_ACCEL_SMALL_MS2));
    return taxi + rollBundleSec / 60;
  }
  
  function getNormalizedStandDwellBounds(f) {
    let dwell = f.dwellMin != null ? f.dwellMin : 0;
    let minDwell = f.minDwellMin != null ? f.minDwellMin : 0;
    dwell = Math.max(SCHED_DWELL_FLOOR_MIN, dwell);
    minDwell = Math.max(SCHED_DWELL_FLOOR_MIN, minDwell);
    if (minDwell > dwell) minDwell = dwell;
    return { dwell, minDwell };
  }

  /**
   * Apron Gantt SIBT handle: if dwell can shrink (dwell > minDwell), fix SOBT at drag anchor and resize dwell;
   * EIBT shifts by the same Δ as SIBT. If already at min dwell, translate the S block and nudge EOBT/ETOT by Δ.
   */
  function _ganttApplySibtHandleSnappedMinutes(f, mSnapped, dragCtx) {
    if (!f || !dragCtx || flightBlockedLikeNoWay(f)) return false;
    const mClamped = Math.max(0, Number(mSnapped));
    if (!isFinite(mClamped)) return false;
    const anchor = dragCtx.anchorSobt;
    const startS = dragCtx.startSibt;
    const minD = dragCtx.minDwell0;
    const d0 = dragCtx.dwell0;
    if (!(typeof anchor === 'number' && isFinite(anchor)) || !(typeof startS === 'number' && isFinite(startS))) return false;
    const atMinDwell = !(d0 > minD + 1e-9);
    if (atMinDwell) {
      if (typeof applyScheduledGateTimingFromSField === 'function') applyScheduledGateTimingFromSField(f, 'sibt', mClamped);
      const ds = mClamped - startS;
      if (dragCtx.startEobt != null && isFinite(dragCtx.startEobt)) f.eobtMin = dragCtx.startEobt + ds;
      if (dragCtx.startEtot != null && isFinite(dragCtx.startEtot)) f.etotMin = dragCtx.startEtot + ds;
      return true;
    }
    let newDwell = anchor - mClamped;
    let sibtU = mClamped;
    if (newDwell < minD) {
      newDwell = minD;
      sibtU = anchor - minD;
    }
    f.timeMin = sibtU;
    f.sibtMin = sibtU;
    f.sldtMin = Math.max(0, sibtU - SCHED_SIBT_MINUS_SLDT_MIN);
    f.sobtMin = anchor;
    f.dwellMin = Math.max(SCHED_DWELL_FLOOR_MIN, newDwell);
    if (f.minDwellMin != null) {
      f.minDwellMin = Math.max(SCHED_DWELL_FLOOR_MIN, Math.min(f.dwellMin, f.minDwellMin));
    }
    f.stotMin = scheduledStotFromSobtMinutes(f, anchor);
    const deibt = sibtU - startS;
    if (dragCtx.startEibt != null && isFinite(dragCtx.startEibt)) f.eibtMin = dragCtx.startEibt + deibt;
    return true;
  }

  function applyForwardEobtEtotAndDepTaxiDelay(f, eibtMin, etotRunwayCandidateMin) {
    if (!f) return;
    const eibt = eibtMin != null && isFinite(eibtMin) ? eibtMin : 0;
    const block = (typeof getDepBlockOutMin === 'function') ? getDepBlockOutMin(f) : 0;
    const { dwell, minDwell } = getNormalizedStandDwellBounds(f);
    const low = eibt + minDwell;
    const high = eibt + dwell;
    const sobtPref = (f.sobtMin != null)
      ? f.sobtMin
      : (f.sibtMin != null
        ? f.sibtMin + dwell
        : (f.timeMin != null ? f.timeMin + dwell : low));
    const eobt = Math.min(Math.max(sobtPref, low), high);
    const etotDraft = eobt + block;
    let etot = etotDraft;
    if (etotRunwayCandidateMin != null && isFinite(etotRunwayCandidateMin)) {
      etot = Math.max(etotRunwayCandidateMin, etotDraft);
    }
    f.eobtMin = eobt;
    f.etotMin = etot;
    f.depTaxiDelayMin = Math.max(0, etot - etotDraft);
  }

  function pinEarliestEldtToSldtPerRunway(flights) {
    void flights;
  }

  var __schedRetStatsBatchActive = false;
  var __schedRetStatsCached = null;
  var __schedRetExitDistSig = '';
  var __schedRetExitDistMemo = null;
  function scheduleRetExitDistLayoutSig() {
    const tws = state.taxiways || [];
    const parts = [];
    for (let i = 0; i < tws.length; i++) {
      const t = tws[i];
      if (!t || (t.pathType !== 'runway' && t.pathType !== 'runway_exit')) continue;
      let line = String(t.id) + '\x1e' + String(t.pathType) + '\x1e' + JSON.stringify(t.vertices || []);
      if (t.pathType === 'runway' && typeof getTaxiwayDirection === 'function') {
        line += '\x1e' + String(getTaxiwayDirection(t));
      }
      if (t.pathType === 'runway_exit') {
        line += '\x1e' + JSON.stringify(t.allowedRwDirections || []);
        if (typeof getTaxiwayDirection === 'function') {
          line += '\x1e' + String(getTaxiwayDirection(t));
        }
      }
      parts.push(line);
    }
    parts.sort();
    return parts.join('\x1f') + '\x1e' + 'arrivalRetPathEdgeF1V1';
  }
  function bumpScheduleRetExitDistCache() {
    __schedRetExitDistSig = '';
    __schedRetExitDistMemo = null;
  }
  /** Arrival 표/샘플 공통용: 활주로 CW·CCW(``both``는 CW 대용)만큼만 RET 행 유지(F2 가용 활주선 방향). */
  function filterScheduleRetStatsForArrivalOperationalLayout(raw) {
    if (!Array.isArray(raw)) return [];
    return raw.filter(function(r) {
      if (!r || !r.runway || !r.exit) return false;
      return arrivalRetPassesFilter2RunwayAvailableDir(r.runway, r.exit);
    });
  }
  function beginScheduleRetStatsBatch() {
    __schedRetStatsBatchActive = true;
    __schedRetStatsCached = null;
  }
  function endScheduleRetStatsBatch() {
    __schedRetStatsBatchActive = false;
    if (__schedRetStatsCached != null) {
      const sig = scheduleRetExitDistLayoutSig();
      __schedRetExitDistSig = sig;
      __schedRetExitDistMemo = __schedRetStatsCached;
    }
    __schedRetStatsCached = null;
  }
  function getScheduleRetStatsAll() {
    if (__schedRetStatsBatchActive) {
      if (__schedRetStatsCached === null) {
        const raw = typeof computeRunwayExitDistances === 'function' ? computeRunwayExitDistances() : [];
        __schedRetStatsCached = filterScheduleRetStatsForArrivalOperationalLayout(raw);
      }
      return __schedRetStatsCached;
    }
    const sig = scheduleRetExitDistLayoutSig();
    if (sig === __schedRetExitDistSig && __schedRetExitDistMemo && Array.isArray(__schedRetExitDistMemo)) {
      return __schedRetExitDistMemo;
    }
    const res = typeof computeRunwayExitDistances === 'function' ? computeRunwayExitDistances() : [];
    const filtered = filterScheduleRetStatsForArrivalOperationalLayout(res);
    __schedRetExitDistSig = sig;
    __schedRetExitDistMemo = filtered;
    return filtered;
  }

  function warmFlightPathsForSchedule(flights) {
    void flights;
  }

  function warmPathsEnsureArrRetRot(flights, forceResampleRet) {
    warmFlightPathsForSchedule(flights);
    return (typeof ensureArrRetRotSampled === 'function')
      ? ensureArrRetRotSampled(flights, !!forceResampleRet)
      : getScheduleRetStatsAll();
  }

  function mutRotCfgEntryForType(configByType, f) {
    const ac = typeof getAircraftInfoByType === 'function' ? getAircraftInfoByType(f.aircraftType) : null;
    const typeKey = f.aircraftType || (ac && ac.id) || (ac && ac.name) || '';
    if (!typeKey) return null;
    if (configByType[typeKey]) return configByType[typeKey];
    const tdMu = (typeof ac?.touchdown_zone_avg_m === 'number') ? ac.touchdown_zone_avg_m : 900;
    const vMu = (typeof ac?.touchdown_speed_avg_ms === 'number') ? ac.touchdown_speed_avg_ms : 70;
    const aMu = (typeof ac?.deceleration_avg_ms2 === 'number') ? ac.deceleration_avg_ms2 : 2.5;
    const tdSigma = Math.round(tdMu * 0.1);
    const vSigma = Math.round(vMu * 0.1);
    const aSigma = Math.round(aMu * 0.1 * 10) / 10;
    configByType[typeKey] = { tdMu, tdSigma, vMu, vSigma, aMu, aSigma };
    return configByType[typeKey];
  }
  /** Same runway resolution as graphPathArrival (token.arrRunwayId before generic runwayId). */
  function resolveArrivalRunwayIdForFlight(f) {
    if (!f) return null;
    const t = f.token || {};
    return t.arrRunwayId || t.runwayId || f.arrRunwayId || null;
  }
  function isValidSampledArrRetForFlight(f, retStatsAll) {
    if (!f || f.sampledArrRet == null) return false;
    if (!Array.isArray(retStatsAll) || !retStatsAll.length) return false;
    const arrRunwayId = resolveArrivalRunwayIdForFlight(f);
    return retStatsAll.some(function(r) {
      if (!r || !r.exit || r.exit.id !== f.sampledArrRet) return false;
      if (arrRunwayId == null) return true;
      return !!(r.runway && r.runway.id === arrRunwayId);
    });
  }
  /** Runway-exit (RET) sampling for Arrival Configuration / schedule RET column. ROT(arr) seconds come from Pro Sim schedule (``ARR_ROT_SEC``), not from this function. */
  function sampleArrRetRotForFlightIfNeeded(f, retStatsAll, configByType, forceResample) {
    if (!f) return;
    const rev = state.vttArrCacheRev | 0;
    if (!forceResample && f.timeline_meta && typeof f.timeline_meta === 'object' &&
        f.timeline_meta.playbackSource === 'des_result') {
      f.__schedRetRotRev = rev;
      return;
    }
    if (!forceResample) return;
    const arrRunwayId = resolveArrivalRunwayIdForFlight(f);
    const cfg = mutRotCfgEntryForType(configByType, f);
    if (!cfg || !retStatsAll || !retStatsAll.length || arrRunwayId == null) {
      f.__schedRetRotRev = rev;
      return;
    }
    const rwObj = resolveArrivalRunwayTaxiwayFromState(arrRunwayId);
    const slotIdx = arrivalPathOpsSlotIndexFromFlightSimAnchor(f);
    const icaoLetter = flightIcaoLetterForArrivalInfra(f);
    if (!icaoLetter || !rwObj) {
      f.sampledArrRet = null;
      f.arrRetFailed = true;
      f.arrDecelMs2 = null;
      f.__schedRetRotRev = rev;
      return;
    }
    if (pathOpsBlockedOpenOrIcaoAtSlot(rwObj, slotIdx, icaoLetter)) {
      f.sampledArrRet = null;
      f.arrRetFailed = true;
      f.arrDecelMs2 = null;
      f.__schedRetRotRev = rev;
      return;
    }
    const effDir = arrivalEffectiveRunwayDirForSlot(rwObj, slotIdx);
    if (!effDir || (effDir !== 'clockwise' && effDir !== 'counter_clockwise')) {
      f.sampledArrRet = null;
      f.arrRetFailed = true;
      f.arrDecelMs2 = null;
      f.__schedRetRotRev = rev;
      return;
    }
    const rdLayout = getRunwayOperationalDirForArrivalRetFilter2(rwObj);
    const minArrVelRwy = getMinArrVelocityMpsForRunwayId(arrRunwayId);
    const tdSample = sampleNormal(cfg.tdMu, cfg.tdSigma);
    const tdMin = cfg.tdMu * 0.85;
    const tdMax = cfg.tdMu * 1.15;
    const dTd = clamp(tdSample, Math.max(0, tdMin), Math.max(0, tdMax));
    const vSample = sampleNormal(cfg.vMu, cfg.vSigma);
    const vMin = cfg.vMu * 0.85;
    const vMax = cfg.vMu * 1.15;
    const v0 = clamp(vSample, Math.max(0, vMin), Math.max(0, vMax));
    const aSample = sampleNormal(cfg.aMu, cfg.aSigma);
    const aMin = Math.max(0.1, cfg.aMu * 0.85);
    const aMax = Math.min(6, cfg.aMu * 1.15);
    const aDec = clamp(aSample, aMin, aMax);
    const candidates = retStatsAll.filter(function(r) {
      if (!(r && r.runway && r.runway.id === arrRunwayId && r.exit)) return false;
      if (!arrivalRetPassesFilter2RunwayAvailableDir(r.runway, r.exit, rdLayout)) return false;
      const ex = r.exit;
      if (pathOpsBlockedOpenOrIcaoAtSlot(ex, slotIdx, icaoLetter)) return false;
      if (!pathOpsRetCwCcwBranchOpenAtSlot(ex, slotIdx, effDir)) return false;
      return true;
    });
    if (!candidates.length) {
      f.sampledArrRet = null;
      f.arrRetFailed = true;
      f.arrDecelMs2 = null;
      f.__schedRetRotRev = rev;
      return;
    }
    let chosen = null;
    candidates.forEach(r => {
      if (chosen) return;
      const distFromTd = Math.max(0, r.distM - dTd);
      const vAt = runwayArrSpeedAndTimeToRet(v0, aDec, distFromTd, minArrVelRwy).vAtRet;
      if (vAt <= r.maxExitVelocity) { chosen = r; }
    });
    if (chosen) {
      f.sampledArrRet = chosen.exit && chosen.exit.id || null;
      f.arrRetFailed = false;
      const MAX_DECEL_MS2 = 15;
      const distFromTdChosen = Math.max(0, chosen.distM - dTd);
      const aDecRot = Math.min(aDec, MAX_DECEL_MS2);
      const rtRunway = runwayArrSpeedAndTimeToRet(v0, aDecRot, distFromTdChosen, minArrVelRwy);
      const vAtChosen = rtRunway.vAtRet;
      const minExitVel = (typeof chosen.minExitVelocity === 'number' && isFinite(chosen.minExitVelocity) && chosen.minExitVelocity > 0)
        ? Math.min(chosen.minExitVelocity, chosen.maxExitVelocity || chosen.minExitVelocity)
        : 15;
      f.arrRunwayIdUsed = arrRunwayId;
      f.arrTdDistM = dTd;
      f.arrRetDistM = chosen.distM;
      f.arrVTdMs = v0;
      f.arrVRetInMs = vAtChosen;
      f.arrVRetOutMs = minExitVel;
      f.arrDecelMs2 = aDecRot;
    } else {
      f.sampledArrRet = null;
      f.arrRetFailed = true;
      f.arrDecelMs2 = null;
    }
    f.__schedRetRotRev = rev;
  }
  function ensureArrRetRotSampled(flights, forceResampleRet) {
    const retStatsAll = (typeof getScheduleRetStatsAll === 'function') ? getScheduleRetStatsAll() : [];
    if (!Array.isArray(flights) || !flights.length) return retStatsAll;
    if (!forceResampleRet) return retStatsAll;
    const configByType = {};
    flights.forEach(f => { mutRotCfgEntryForType(configByType, f); });
    flights.forEach(function(f) {
