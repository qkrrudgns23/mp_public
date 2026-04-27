  function getFlightPoseAtTimeForDraw(flight, tSec) {
    const tl = flight && flight.timeline;
    if (!tl || !tl.length) return null;
    let t = Number(tSec);
    if (!isFinite(t)) return null;
    const t0 = tl[0].t, t1 = tl[tl.length - 1].t;
    if (t + 1e-9 < t0) return null;
    if (t > t1) t = t1;
    return applyEobtApronDepTaxiPushbackNoseIfNeeded(flight, t, getFlightPoseAtTime(flight, t));
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
    const tl = f && f.timeline;
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
    const tl = f && f.timeline;
    if (!tl || tl.length < 2 || !(maxDistM > 0)) return [];
    const tMin = tl[0].t, tMax = tl[tl.length - 1].t;
    let t = Math.min(Math.max(tEnd, tMin), tMax);
    let seg = 0;
    for (let i = 0; i < tl.length - 1; i++) {
      if (t >= tl[i].t && t <= tl[i + 1].t) { seg = i; break; }
      if (t > tl[i + 1].t) seg = i;
    }
    const pts = [];
    function xyAt(T) {
      if (T <= tMin) return [tl[0].x, tl[0].y];
      if (T >= tMax) return [tl[tl.length - 1].x, tl[tl.length - 1].y];
      for (let i = 0; i < tl.length - 1; i++) {
        const a = tl[i], b = tl[i + 1];
        if (T >= a.t && T <= b.t) {
          const sp = b.t - a.t || 1;
          const uu = (T - a.t) / sp;
          return [a.x + (b.x - a.x) * uu, a.y + (b.y - a.y) * uu];
        }
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
    else if (role === 'term') el.value = f.terminalId || (f.token && f.token.terminalId) || '';
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
      const tid = f.terminalId || (f.token && f.token.terminalId) || '';
      return 'b:' + (tid ? getTerminalDisplayLabelById(tid) : '—');
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
    const termEl = document.getElementById('flightAssignStripTerm');
    const depEl = document.getElementById('flightAssignStripDep');
    const intDomEl = document.getElementById('flightAssignStripIntDom');
    if (arrEl) {
      const sid = f ? (resolveArrivalRunwayIdForFlight(f) || '') : '';
      arrEl.innerHTML = buildRunwayOptionsHtml(sid);
      arrEl.value = sid;
    }
    if (intDomEl) {
      intDomEl.value = (f && String(f.intDom || '').trim().toLowerCase() === 'dom') ? 'Dom' : 'Int';
    }
    if (termEl) {
      const tid = f ? (f.terminalId || (f.token && f.token.terminalId) || '') : '';
      termEl.innerHTML = buildTerminalOptionsHtml(tid);
      termEl.value = tid;
    }
    if (depEl) {
      const did = f ? (f.depRunwayId || (f.token && f.token.depRunwayId) || '') : '';
      depEl.innerHTML = buildRunwayOptionsHtml(did);
      depEl.value = did;
    }
  }
  function syncFlightAssignStrip() {
    const arrEl = document.getElementById('flightAssignStripArr');
    const termEl = document.getElementById('flightAssignStripTerm');
    const depEl = document.getElementById('flightAssignStripDep');
    const intDomEl = document.getElementById('flightAssignStripIntDom');
    const sel = state.selectedObject;
    const hasFlight = sel && sel.type === 'flight' && sel.id;
    const f = hasFlight ? state.flights.find(function(x) { return x.id === sel.id; }) : null;
    const dis = !f;
    [arrEl, termEl, depEl, intDomEl].forEach(function(el) {
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
    } else if (role === 'term') {
      const r = resolveTerminalIdFromInput(raw);
      if ((raw || '').trim() && r === undefined) {
        syncFlightAssignStripFromFlight(f);
        return;
      }
      val = r === undefined ? null : r;
    } else return;
    var prevArr = f.arrRunwayId || null;
    var prevDep = f.depRunwayId || (f.token && f.token.depRunwayId) || null;
    var prevTerm = f.terminalId || (f.token && f.token.terminalId) || null;
    if (role === 'arr' && val === prevArr) return;
    if (role === 'dep' && val === prevDep) return;
    if (role === 'term' && val === prevTerm) return;
    var prevStand = f.standId || null;
    if (!f.token) f.token = { nodes: ['runway','taxiway','apron','terminal'], runwayId: null, apronId: null, terminalId: null };
    if (role === 'arr') {
      f.arrRunwayId = val;
      f.token.runwayId = val;
    } else if (role === 'term') {
      f.terminalId = val;
      f.token.terminalId = val;
      if (f.standId) {
        var allStands = (st.pbbStands || []).concat(st.remoteStands || []).concat(st.tempStands || []);
        var stand = allStands.find(function(s) { return s.id === f.standId; });
        if (stand) {
          var term = getTerminalForStand(stand);
          var standTermId = term ? term.id : null;
          if (!val || !standTermId || val !== standTermId) f.standId = null;
        }
      }
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

  const FLIGHT_SCHED_TABLE_COL_COUNT = 24;
  /** tbody td index alignment with `_buildFlightListRowHtml` (0-based). SLDT(d)/STOT(d) hidden from table. */
  const FLIGHT_SCHED_TD_SIBTD = 13;
  const FLIGHT_SCHED_TD_SOBTD = 14;
  const FLIGHT_SCHED_TD_ELDT = 15;
  const FLIGHT_SCHED_TD_EIBT = 16;
  const FLIGHT_SCHED_TD_EOBT = 17;
  const FLIGHT_SCHED_TD_ETOT = 18;
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
    parts.push('<tr class=\"flight-virt-spacer\" aria-hidden=\"true\" style=\"height:' + topPad + 'px\"><td colspan=\"' + FLIGHT_SCHED_TABLE_COL_COUNT + '\"></td></tr>');
    for (let i = start; i < end; i++) {
      parts.push(_buildFlightListRowHtml(flightsSorted[i], retStatsAll));
    }
    parts.push('<tr class=\"flight-virt-spacer\" aria-hidden=\"true\" style=\"height:' + botPad + 'px\"><td colspan=\"' + FLIGHT_SCHED_TABLE_COL_COUNT + '\"></td></tr>');
    tbody.innerHTML = parts.join('');
    _flightListWireEvents(listEl, state);
  }
  function _flightListTeardownVirtual(listEl) {
    listEl._flightVirtState = null;
  }
  function _flightListMountVirtual(listEl, flightsSorted, retStatsAll, headerRow) {
    const prevScroll = listEl.querySelector('.flight-schedule-table[data-virtual-table=\"1\"]') ? (listEl.scrollTop || 0) : 0;
    listEl._flightVirtState = {
      flightsSorted: flightsSorted,
      retStatsAll: retStatsAll,
      rowH: DOM_OPT_FLIGHT_VIRT_ROW_H,
      overscan: DOM_OPT_FLIGHT_VIRT_OVERSCAN,
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
  
  function applyForwardEobtEtotAndDepTaxiDelay(f, eibtMin, etotRunwayCandidateMin) {
    if (!f) return;
    const eibt = eibtMin != null && isFinite(eibtMin) ? eibtMin : 0;
    const block = (typeof getDepBlockOutMin === 'function') ? getDepBlockOutMin(f) : 0;
    const { dwell, minDwell } = getNormalizedStandDwellBounds(f);
    const low = eibt + minDwell;
    const high = eibt + dwell;
    const sobtPref = (f.sobtMin_d != null)
      ? f.sobtMin_d
      : (f.sibtMin_d != null
        ? f.sibtMin_d + dwell
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
        __schedRetStatsCached = typeof computeRunwayExitDistances === 'function' ? computeRunwayExitDistances() : [];
      }
      return __schedRetStatsCached;
    }
    const sig = scheduleRetExitDistLayoutSig();
    if (sig === __schedRetExitDistSig && __schedRetExitDistMemo && Array.isArray(__schedRetExitDistMemo)) {
      return __schedRetExitDistMemo;
    }
    const res = typeof computeRunwayExitDistances === 'function' ? computeRunwayExitDistances() : [];
    __schedRetExitDistSig = sig;
    __schedRetExitDistMemo = res;
    return res;
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
    if (!forceResample && f.__schedRetRotRev === rev && isValidSampledArrRetForFlight(f, retStatsAll)) return;
    if (!forceResample && (f.__schedRetRotRev === undefined || f.__schedRetRotRev === null) &&
        f.sampledArrRet != null && f.arrRetFailed === false &&
        isValidSampledArrRetForFlight(f, retStatsAll)) {
      f.__schedRetRotRev = rev;
      return;
    }
    if (f.sampledArrRet != null && !isValidSampledArrRetForFlight(f, retStatsAll)) {
      f.sampledArrRet = null;
      f.arrRetFailed = false;
      f.arrDecelMs2 = null;
    }
    const arrRunwayId = resolveArrivalRunwayIdForFlight(f);
    const cfg = mutRotCfgEntryForType(configByType, f);
    if (!cfg || !retStatsAll || !retStatsAll.length || arrRunwayId == null) {
      f.__schedRetRotRev = rev;
      return;
    }
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
    const aMax = Math.min(6,   cfg.aMu * 1.15);
    const aDec = clamp(aSample, aMin, aMax);
    const candidates = retStatsAll.filter(function(r) {
      return !!(r && r.runway && r.runway.id === arrRunwayId && r.exit);
    });
    if (!candidates.length) {
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
    if (!Array.isArray(flights) || !flights.length) return [];
    const configByType = {};
    flights.forEach(f => { mutRotCfgEntryForType(configByType, f); });
    const retStatsAll = getScheduleRetStatsAll();
    flights.forEach(function(f) {
      sampleArrRetRotForFlightIfNeeded(f, retStatsAll, configByType, !!forceResampleRet);
    });
    return retStatsAll;
  }

  function _flightListEmptyHtml(message) {
    return '<div style="font-size:11px;color:#9ca3af;">' + message + '</div>';
  }

  function _renderEmptyFlightListState(listEl, cfgEl) {
    state.flightSchedulePage = 0;
    const pgr = document.getElementById('flightSchedulePager');
    if (pgr) pgr.style.display = 'none';
    _flightListTeardownVirtual(listEl);
    listEl.innerHTML = _flightListEmptyHtml('No flights yet.');
    if (cfgEl) cfgEl.innerHTML = _flightListEmptyHtml('No flights yet.');
    const ganttEl = document.getElementById('allocationGantt');
    if (ganttEl) ganttEl.innerHTML = _flightListEmptyHtml('No flights for Gantt.');
    if (typeof ensureFlightAssignStripWired === 'function') ensureFlightAssignStripWired();
    if (typeof syncFlightAssignStrip === 'function') syncFlightAssignStrip();
  }
  function _updateFlightSchedulePagerUI(totalCount) {
    const pager = document.getElementById('flightSchedulePager');
    if (!pager) return;
    const size = FLIGHT_SCHED_PAGE_SIZE;
    if (!size || size <= 0) {
      pager.style.display = 'none';
      return;
    }
    pager.style.display = 'flex';
    const maxPage = Math.max(0, Math.ceil(totalCount / size) - 1);
    if (state.flightSchedulePage > maxPage) state.flightSchedulePage = maxPage;
    if (state.flightSchedulePage < 0) state.flightSchedulePage = 0;
    const start = state.flightSchedulePage * size;
    const end = Math.min(totalCount, start + size);
    const pageNum = maxPage + 1;
    const cur = state.flightSchedulePage + 1;
    const tEl = document.getElementById('flightSchedulePagerTotal');
    const rEl = document.getElementById('flightSchedulePagerRange');
    if (tEl) tEl.textContent = String(totalCount);
    if (rEl) rEl.textContent = totalCount ? (String(start + 1) + '–' + String(end) + ' · p ' + String(cur) + '/' + String(pageNum)) : '0–0 · p 0/0';
    const bPrev = document.getElementById('btnFlightSchedPrev');
    const bNext = document.getElementById('btnFlightSchedNext');
    if (bPrev) bPrev.disabled = state.flightSchedulePage <= 0;
    if (bNext) bNext.disabled = state.flightSchedulePage >= maxPage;
  }

  /** Same predicate as Arrival Configuration "Failed" row (flight-ui _renderFlightConfigTable failedCounts). */
  function isFlightArrRetFailedInConfigTable(f, retStatsAll) {
    if (!f) return false;
    if (!Array.isArray(retStatsAll) || !retStatsAll.length) return false;
    return f.sampledArrRet === null || typeof f.sampledArrRet === 'undefined';
  }
  function arrivalConfigColumnKeyForFlight(f) {
    if (!f) return '';
    const ac = typeof getAircraftInfoByType === 'function' ? getAircraftInfoByType(f.aircraftType) : null;
    return f.aircraftType || (ac && ac.id) || (ac && ac.name) || '';
  }
  function isFlightCountedInArrivalConfigFailedRow(f, retStats) {
    return isFlightArrRetFailedInConfigTable(f, retStats) && !!arrivalConfigColumnKeyForFlight(f);
  }

  function flightScheduleMinutesForRow(f) {
    const tArrMin = f.timeMin != null ? f.timeMin : 0;
    const dwell = f.dwellMin != null ? f.dwellMin : 0;
    const tDepMin = tArrMin + dwell;
    const sibtD = f.sibtMin_d != null ? f.sibtMin_d : tArrMin;
    const sobtD = f.sobtMin_d != null ? f.sobtMin_d : tDepMin;
    const sldtCalc = (f.sldtMin_d != null ? f.sldtMin_d : Math.max(0, sibtD - SCHED_SD_SIBT_MINUS_SLD_MIN));
    const sldtOrig = f.sldtMin_orig != null ? f.sldtMin_orig : sldtCalc;
    const sobtOrig = (f.sobtMin_orig != null) ? f.sobtMin_orig : tDepMin;
    const stotOrig = (f.stotMin_orig != null) ? f.stotMin_orig : (sobtD + SCHED_SD_STOT_PLUS_SOBD_MIN);
    return {
      sibt: tArrMin,
      sobt: tDepMin,
      sldt_d: f.sldtMin_d != null ? f.sldtMin_d : sldtOrig,
      sibt_d: f.sibtMin_d != null ? f.sibtMin_d : tArrMin,
      sobt_d: f.sobtMin_d != null ? f.sobtMin_d : tDepMin,
      stot_d: f.stotMin_d != null ? f.stotMin_d : stotOrig,
    };
  }

  function flightScheduleProSimTimedCell(displayStr, dotKind) {
    const d = '—';
    const has = displayStr != null && String(displayStr).trim() !== '' && displayStr !== d;
    const text = has ? String(displayStr) : d;
    const muted = has ? '' : ' flight-sched-dot--muted';
    let dotClass = 'flight-sched-dot--green';
    if (dotKind === 'vttarr') dotClass = 'flight-sched-dot--vttarr';
    else if (dotKind === 'red') dotClass = 'flight-sched-dot--red';
    else if (dotKind === 'pink') dotClass = 'flight-sched-dot--pink';
    return '<span class="flight-sched-cell-inner">' +
      '<span class="flight-sched-dot ' + dotClass + muted + '" aria-hidden="true"></span>' +
      '<span class="flight-sched-cell-text">' + (has ? escapeHtml(text) : d) + '</span></span>';
  }

  function _buildFlightListHeaderHtml() {
    return '' +
      '<table class="flight-schedule-table">' +
      '<thead><tr>' +
        '<th>Reg</th>' +
        '<th class="flight-th-mixed">Airline</th>' +
        '<th class="flight-th-mixed">Flight Num</th>' +
        '<th>Arr Rw</th>' +
        '<th>Arr RET</th>' +
        '<th>Building</th>' +
        '<th>Apron</th>' +
        '<th>Dep Rw</th>' +
        '<th class="flight-col-s flight-col-s-start flight-td-sibt">SIBT</th>' +
        '<th class="flight-col-s flight-col-s-last">SOBT</th>' +
        '<th class="flight-col-sd flight-col-sd-start">SIBT(d)</th>' +
        '<th class="flight-col-sd flight-col-sd-last">SOBT(d)</th>' +
        '<th class="flight-col-e flight-col-e-start">ELDT</th>' +
        '<th class="flight-col-e">EIBT</th>' +
        '<th class="flight-col-e">EOBT</th>' +
        '<th class="flight-col-e">ETOT</th>' +
        '<th class="flight-col-e flight-col-rot flight-th-mixed">ROT(arr)</th>' +
        '<th class="flight-th-mixed">VTT(Arr)</th>' +
        '<th class="flight-th-mixed">ATT(Dep)</th>' +
        '<th class="flight-col-e flight-col-rot flight-th-mixed">ROT(dep)</th>' +
        '<th class="flight-td-del"></th>' +
      '</tr></thead>' +
      '<tbody>';
  }

  function _buildFlightListRowHtml(f, retStatsAll) {
    const arrRunwayId = resolveArrivalRunwayIdForFlight(f);
    const ac = typeof getAircraftInfoByType === 'function' ? getAircraftInfoByType(f.aircraftType) : null;
    const arrRetFailed = isFlightCountedInArrivalConfigFailedRow(f, retStatsAll);
    let sampledRetName = '—';
    if (arrRetFailed) sampledRetName = 'Failed';
    else if (f.sampledArrRet != null && retStatsAll && retStatsAll.length) {
      const retInfo = retStatsAll.find(r => r.exit && r.exit.id === f.sampledArrRet);
      sampledRetName = retInfo ? (retInfo.name || 'RET') : 'RET';
    }
    const tArrMin = f.timeMin != null ? f.timeMin : 0;
    const dwell = f.dwellMin != null ? f.dwellMin : 0;
    const tDepMin = tArrMin + dwell;
    const schedDepRotMin = Math.max(0, Number(SCHED_DEP_ROT_MIN) || 2);
    const sldtCalc = (f.sldtMin_d != null ? f.sldtMin_d : Math.max(0, tArrMin));
    const sldtOrig = f.sldtMin_orig != null ? f.sldtMin_orig : sldtCalc;
    const sobtOrig = (f.sobtMin_orig != null) ? f.sobtMin_orig : tDepMin;
    const stotOrig = (f.stotMin_orig != null) ? f.stotMin_orig : (tDepMin + schedDepRotMin);
    if (f.sobtMin_orig == null) {
      f.sldtMin_orig = sldtOrig;
      f.sibtMin_orig = tArrMin;
      f.sobtMin_orig = sobtOrig;
      f.stotMin_orig = stotOrig;
    }
    const schedM = flightScheduleMinutesForRow(f);
    const sibtDisp = formatFlightScheduleDateTime(f, schedM.sibt);
    const sobtDisp = formatFlightScheduleDateTime(f, schedM.sobt);
    const sibtStr_d = formatFlightScheduleDateTime(f, schedM.sibt_d);
    const sobtStr_d = formatFlightScheduleDateTime(f, schedM.sobt_d);
    function fmtFlightESchedCell(minVal) {
      if (!(typeof minVal === 'number' && isFinite(minVal))) return '—';
      return formatFlightScheduleDateTime(f, minVal);
    }
    const eldtStr = fmtFlightESchedCell(f.eldtMin);
    const eibtStr = fmtFlightESchedCell(f.eibtMin);
    const eobtStr = fmtFlightESchedCell(f.eobtMin);
    const etotStr = fmtFlightESchedCell(f.etotMin);
    const dash = '—';
    const rotArrStr = (f.arrRotSec != null && isFinite(f.arrRotSec)) ? formatSecondsToHHMMSS(f.arrRotSec) : dash;
    const vttArrStr = (f.proSimVttArrSec != null && isFinite(f.proSimVttArrSec)) ? formatSecondsToHHMMSS(f.proSimVttArrSec) : dash;
    const attDepStr = (f.proSimVttDepSec != null && isFinite(f.proSimVttDepSec)) ? formatSecondsToHHMMSS(f.proSimVttDepSec) : dash;
    const rotDepStr = (f.proSimDepLineupSec != null && isFinite(f.proSimDepLineupSec)) ? formatSecondsToHHMMSS(f.proSimDepLineupSec) : dash;
    const rotArrCell = flightScheduleProSimTimedCell(rotArrStr, 'green');
    const vttArrCell = flightScheduleProSimTimedCell(vttArrStr, 'vttarr');
    const attDepCell = flightScheduleProSimTimedCell(attDepStr, 'red');
    const rotDepCell = flightScheduleProSimTimedCell(rotDepStr, 'pink');
    const depRunwayId = f.depRunwayId || (f.token && f.token.depRunwayId);
    const termId = f.terminalId || (f.token && f.token.terminalId);
    const arrRwRead = escapeHtml(getRunwayDisplayLabelById(arrRunwayId));
    const buildingRead = escapeHtml(getTerminalDisplayLabelById(termId));
    const depRwRead = escapeHtml(getRunwayDisplayLabelById(depRunwayId));
    const aircraftTypeLabel = ac ? (ac.name || ac.id || '') : (f.aircraftType || '—');
    const codeIcao = (ac && ac.icao) ? ac.icao : (f.code || '—');
    const intDomVal = (String(f.intDom || '').trim().toLowerCase() === 'dom') ? 'Dom' : 'Int';
    return '' +
      '<tr class="flight-data-row obj-item" data-id="' + f.id + '">' +
        '<td class="flight-td-reg">' + escapeHtml(f.reg || '') + '</td>' +
        '<td class="flight-td-reg">' + escapeHtml(f.airlineCode || '') + '</td>' +
        '<td class="flight-td-reg">' + escapeHtml(f.flightNumber || '') + '</td>' +
        '<td>' + escapeHtml(aircraftTypeLabel) + '</td>' +
        '<td>' + escapeHtml(codeIcao) + '</td>' +
        '<td class="flight-td-readonly" title="Edit in Int/Dom above when flight is selected">' + escapeHtml(intDomVal) + '</td>' +
        '<td class="flight-td-readonly">' + arrRwRead + '</td>' +
        '<td class="flight-td-arr-ret' + (arrRetFailed ? ' flight-td-arr-ret-failed' : '') + '">' + (arrRetFailed ? 'Failed' : escapeHtml(sampledRetName)) + '</td>' +
        '<td class="flight-td-readonly">' + buildingRead + '</td>' +
        '<td class="flight-td-reg">' + (function() { var st = findStandById(f.standId); return escapeHtml(st ? ((st.name && st.name.trim()) || st.id || '—') : '—'); })() + '</td>' +
        '<td class="flight-td-readonly">' + depRwRead + '</td>' +
        '<td class="flight-td-time flight-col-s flight-col-s-start flight-td-sibt" data-sched-min="' + schedM.sibt + '">' + escapeHtml(sibtDisp) + '</td>' +
        '<td class="flight-td-time flight-col-s flight-col-s-last" data-sched-min="' + schedM.sobt + '">' + escapeHtml(sobtDisp) + '</td>' +
        '<td class="flight-td-time flight-col-sd flight-col-sd-start" data-sched-min="' + schedM.sibt_d + '">' + escapeHtml(sibtStr_d) + '</td>' +
        '<td class="flight-td-time flight-col-sd flight-col-sd-last" data-sched-min="' + schedM.sobt_d + '">' + escapeHtml(sobtStr_d) + '</td>' +
        '<td class="flight-td-time flight-col-e flight-col-e-start">' + escapeHtml(eldtStr) + '</td>' +
        '<td class="flight-td-time flight-col-e">' + escapeHtml(eibtStr) + '</td>' +
        '<td class="flight-td-time flight-col-e">' + escapeHtml(eobtStr) + '</td>' +
        '<td class="flight-td-time flight-col-e">' + escapeHtml(etotStr) + '</td>' +
        '<td class="flight-td-time flight-col-e flight-col-rot">' + rotArrCell + '</td>' +
        '<td class="flight-td-time">' + vttArrCell + '</td>' +
        '<td class="flight-td-time">' + attDepCell + '</td>' +
        '<td class="flight-td-time flight-col-e flight-col-rot">' + rotDepCell + '</td>' +
        '<td class="flight-td-del"><button type="button" class="obj-item-delete" data-del="' + f.id + '">×</button></td>' +
      '</tr>';
  }

  function _buildFlightListRowsHtml(flightsSorted, retStatsAll) {
    return flightsSorted.map(function(f) {
      return _buildFlightListRowHtml(f, retStatsAll);
    });
  }

  const FLIGHT_LIST_PATH_YIELD_CHUNK = 6;
  const FLIGHT_LIST_ASYNC_PATH_MIN = 8;
  function _renderFlightListDomAndSchedule(flightsSorted, schedFull, dirtySet, standSet, listEl, cfgEl, retStatsAll, domOpt) {
    const skipGanttRefresh = domOpt && domOpt.skipGanttRefresh;
    const headerRow = _buildFlightListHeaderHtml();
    const dirtyIds = [];
    dirtySet.forEach(function(id) { if (id != null && id !== '') dirtyIds.push(id); });
    const deferOnlyDirty = false;
    if (schedFull) {
      if (typeof computeScheduledDisplayTimes === 'function') computeScheduledDisplayTimes(state.flights);
    } else {
      if (!deferOnlyDirty && typeof computeScheduledDisplayTimesIncremental === 'function')
        computeScheduledDisplayTimesIncremental(state.flights, dirtySet, standSet);
    }
    flightsSorted.sort((a, b) => (a.sibtMin_d != null ? a.sibtMin_d : (a.timeMin != null ? a.timeMin : 0)) - (b.sibtMin_d != null ? b.sibtMin_d : (b.timeMin != null ? b.timeMin : 0)));
    const usePagination = FLIGHT_SCHED_PAGE_SIZE > 0;
    let flightsForDom = flightsSorted;
    if (usePagination) {
      const size = FLIGHT_SCHED_PAGE_SIZE;
      const n = flightsSorted.length;
      const maxPage = Math.max(0, Math.ceil(n / size) - 1);
      if (state.flightSchedulePage > maxPage) state.flightSchedulePage = maxPage;
      if (state.flightSchedulePage < 0) state.flightSchedulePage = 0;


      const start = state.flightSchedulePage * size;
      flightsForDom = flightsSorted.slice(start, start + size);
    }
    _updateFlightSchedulePagerUI(flightsSorted.length);
    const useVirt = !usePagination && DOM_OPT_FLIGHT_VIRT_ENABLE && flightsSorted.length >= DOM_OPT_FLIGHT_VIRT_MIN;
    if (useVirt) {
      _flightListMountVirtual(listEl, flightsSorted, retStatsAll, headerRow);
    } else {
      _flightListTeardownVirtual(listEl);
      const dataRows = _buildFlightListRowsHtml(flightsForDom, retStatsAll);
      listEl.innerHTML = headerRow + dataRows.join('') + '</tbody></table>';
      const tbl0 = listEl.querySelector('.flight-schedule-table');
      if (tbl0) {
        if (usePagination) tbl0.setAttribute('data-virtual-table', '1');
        else tbl0.removeAttribute('data-virtual-table');
      }
      _flightListWireEvents(listEl, state);
    }
    _renderFlightConfigTable(cfgEl, flightsSorted);
    if (typeof ensureFlightAssignStripWired === 'function') ensureFlightAssignStripWired();
    if (typeof syncFlightAssignStrip === 'function') syncFlightAssignStrip();
    if (!skipGanttRefresh && typeof renderFlightGantt === 'function') renderFlightGantt({ skipPathPrep: true });
  }
  function _renderFlightListAfterPathEnsure(flightsSorted, schedFull, forceResampleRet, dirtySet, standSet, listEl, cfgEl, scheduleOpts) {
    if (forceResampleRet && typeof bumpVttArrCacheRev === 'function') bumpVttArrCacheRev();
    let retStatsAll = [];
    if (schedFull) {
      retStatsAll = (typeof ensureArrRetRotSampled === 'function')
        ? ensureArrRetRotSampled(flightsSorted, !!forceResampleRet)
        : (typeof getScheduleRetStatsAll === 'function' ? getScheduleRetStatsAll() : ((typeof computeRunwayExitDistances === 'function') ? computeRunwayExitDistances() : []));
    } else {
      const dirtyFlights = flightsSorted.filter(function(f) { return dirtySet.has(f.id); });
      const dirtyForRet = dirtyFlights.filter(function(f) { return f; });
      if (dirtyForRet.length && typeof ensureArrRetRotSampled === 'function')
        retStatsAll = ensureArrRetRotSampled(dirtyForRet, false);
      else
        retStatsAll = (typeof getScheduleRetStatsAll === 'function') ? getScheduleRetStatsAll() : ((typeof computeRunwayExitDistances === 'function') ? computeRunwayExitDistances() : []);
    }
    const domOpt = (scheduleOpts && scheduleOpts.skipGanttRefresh) ? { skipGanttRefresh: true } : null;
    _renderFlightListDomAndSchedule(flightsSorted, schedFull, dirtySet, standSet, listEl, cfgEl, retStatsAll, domOpt);
  }

  function renderFlightList(skipAutoAllocate, forceResampleRet, scheduleOpts, onDone) {
    const listEl = document.getElementById('flightList');
    const cfgEl = document.getElementById('flightConfigList');
    const cb = typeof onDone === 'function' ? onDone : null;
    if (!listEl) return;
    if (!state.flights.length) {
      _renderEmptyFlightListState(listEl, cfgEl);
      if (typeof syncProSimButtonFromDesignerPageState === 'function') syncProSimButtonFromDesignerPageState();
      if (cb) cb();
      return;
    }
    if (scheduleOpts && scheduleOpts.pageTurnOnly === true && FLIGHT_SCHED_PAGE_SIZE > 0) {
      const flightsSorted = state.flights.slice();
      flightsSorted.sort((a, b) => (a.sibtMin_d != null ? a.sibtMin_d : (a.timeMin != null ? a.timeMin : 0)) - (b.sibtMin_d != null ? b.sibtMin_d : (b.timeMin != null ? b.timeMin : 0)));
      beginScheduleRetStatsBatch();
      var retStatsAll2 = [];
      try {
        retStatsAll2 = (typeof getScheduleRetStatsAll === 'function')
          ? getScheduleRetStatsAll()
          : ((typeof computeRunwayExitDistances === 'function') ? computeRunwayExitDistances() : []);
        _renderFlightListDomAndSchedule(flightsSorted, false, new Set(), new Set(), listEl, cfgEl, retStatsAll2, { skipGanttRefresh: true });
      } finally {
        endScheduleRetStatsBatch();
      }
      if (typeof syncProSimButtonFromDesignerPageState === 'function') syncProSimButtonFromDesignerPageState();
      if (typeof syncAllocGanttSelectionHighlight === 'function') syncAllocGanttSelectionHighlight();
      if (cb) cb();
      return;
    }
    let schedFull = true;
    let dirtySet = new Set();
    let standSet = new Set();
    if (!forceResampleRet && scheduleOpts && scheduleOpts.scheduleMode === 'incremental') {
      schedFull = false;
      const d = scheduleOpts.dirtyFlightIds;
      if (d instanceof Set) d.forEach(function(id) { if (id != null && id !== '') dirtySet.add(id); });
      else if (Array.isArray(d)) d.forEach(function(id) { if (id != null && id !== '') dirtySet.add(id); });
      const s = scheduleOpts.touchedStandIds;
      if (s instanceof Set) s.forEach(function(id) { if (id != null && id !== '') standSet.add(id); });
      else if (Array.isArray(s)) s.forEach(function(id) { if (id != null && id !== '') standSet.add(id); });
      if (dirtySet.size === 0 && standSet.size === 0) schedFull = true;
    }
    if (forceResampleRet) schedFull = true;
    const flightsSorted = state.flights.slice();
    flightsSorted.sort((a, b) => (a.sibtMin_d != null ? a.sibtMin_d : (a.timeMin != null ? a.timeMin : 0)) - (b.sibtMin_d != null ? b.sibtMin_d : (b.timeMin != null ? b.timeMin : 0)));
    function runTail() {
      beginScheduleRetStatsBatch();
      try {
        _renderFlightListAfterPathEnsure(flightsSorted, schedFull, forceResampleRet, dirtySet, standSet, listEl, cfgEl, scheduleOpts);
        if (typeof syncProSimButtonFromDesignerPageState === 'function') syncProSimButtonFromDesignerPageState();
      } finally {
        endScheduleRetStatsBatch();
      }
      if (cb) cb();
    }
    runTail();
  }

  function _renderFlightConfigTable(cfgEl, flightsSorted) {
    if (!cfgEl) return;
    const seenType = new Set();
    const unique = [];
    flightsSorted.forEach(f => {
      const ac = typeof getAircraftInfoByType === 'function' ? getAircraftInfoByType(f.aircraftType) : null;
      const typeKey = f.aircraftType || (ac && ac.id) || (ac && ac.name) || '';
      if (!typeKey || seenType.has(typeKey)) return;
      seenType.add(typeKey);
      unique.push({
        key: typeKey,
        label: ac ? (ac.name || ac.id || typeKey) : typeKey
      });
    });
    if (!unique.length) {
      cfgEl.innerHTML = _flightListEmptyHtml('No flights yet.');
      return;
    }
    const prevConfigByType = {};
    const prevInputs = cfgEl.querySelectorAll('.flight-config-input[data-ac][data-param]');
    prevInputs.forEach(inp => {
      const acKey = inp.getAttribute('data-ac');
      const param = inp.getAttribute('data-param');
      if (!acKey || !param) return;
      const valNum = Number(inp.value);
      if (!isFinite(valNum)) return;
      if (!prevConfigByType[acKey]) prevConfigByType[acKey] = {};
      prevConfigByType[acKey][param] = valNum;
    });
    const headerCols = unique.map(info => '<th>' + escapeHtml(info.label) + '</th>').join('');
    const cfgHeader = '' +
      '<div style="font-size:10px;color:#9ca3af;margin-bottom:4px;">' +
        'Landing configuration per aircraft type (unit and statistic: mean μ / spread σ).' +
      '</div>' +
      '<table class="flight-schedule-table flight-config-table">' +
      '<thead><tr>' +
        '<th class="sticky-col">Parameter</th>' +
        '<th>Unit</th>' +
        '<th>Stat</th>' +
        headerCols +
      '</tr></thead><tbody>';
    const rows = [];
    const tdMeans = unique.map(info => {
      const acKey = info.key;
      const fromUser = prevConfigByType[acKey] && prevConfigByType[acKey]['td-mean'];
      if (typeof fromUser === 'number' && isFinite(fromUser)) return fromUser;
      const ac = getAircraftInfoByType(acKey) || {};
      return (typeof ac.touchdown_zone_avg_m === 'number') ? ac.touchdown_zone_avg_m : 900;
    });
    const vtdMeans = unique.map(info => {
      const acKey = info.key;
      const fromUser = prevConfigByType[acKey] && prevConfigByType[acKey]['vtd-mean'];
      if (typeof fromUser === 'number' && isFinite(fromUser)) return fromUser;
      const ac = getAircraftInfoByType(acKey) || {};
      return (typeof ac.touchdown_speed_avg_ms === 'number') ? ac.touchdown_speed_avg_ms : 70;
    });
    const aMeans = unique.map(info => {
      const acKey = info.key;
      const fromUser = prevConfigByType[acKey] && prevConfigByType[acKey]['a-mean'];
      if (typeof fromUser === 'number' && isFinite(fromUser)) return fromUser;
      const ac = getAircraftInfoByType(acKey) || {};
      return (typeof ac.deceleration_avg_ms2 === 'number') ? ac.deceleration_avg_ms2 : 2.5;
    });
    const tdSigmas = unique.map((info, idx) => {
      const acKey = info.key;
      const fromUser = prevConfigByType[acKey] && prevConfigByType[acKey]['td-sigma'];
      if (typeof fromUser === 'number' && isFinite(fromUser)) return fromUser;
      const v = tdMeans[idx];
      return Math.round(v * 0.1);
    });
    const vtdSigmas = unique.map((info, idx) => {
      const acKey = info.key;
      const fromUser = prevConfigByType[acKey] && prevConfigByType[acKey]['vtd-sigma'];
      if (typeof fromUser === 'number' && isFinite(fromUser)) return fromUser;
      const v = vtdMeans[idx];
      return Math.round(v * 0.1);
    });
    const aSigmas = unique.map((info, idx) => {
      const acKey = info.key;
      const fromUser = prevConfigByType[acKey] && prevConfigByType[acKey]['a-sigma'];
      if (typeof fromUser === 'number' && isFinite(fromUser)) return fromUser;
      const v = aMeans[idx];
      return Math.round(v * 0.1 * 10) / 10;
    });
    const vTarget = 26;
    const aMeanStopDists = aMeans.map((aMu, idx) => {
      const vMu = vtdMeans[idx];
      const tdMu = tdMeans[idx];
      if (!(aMu > 0) || !(vMu > vTarget)) return Math.max(0, Math.round(tdMu || 0));
      const dFromTouchdown = (vMu*vMu - vTarget*vTarget) / (2 * aMu);
      const dTotal = (tdMu || 0) + (dFromTouchdown > 0 ? dFromTouchdown : 0);
      return dTotal > 0 ? Math.round(dTotal) : 0;
    });

    rows.push(
      '<tr>' +
        '<td class="sticky-col">Touchdown zone distance from threshold</td>' +
        '<td>m</td>' +
        '<td>mean μ</td>' +
        unique.map((info, idx) =>
          '<td><input class="flight-config-input" data-ac="' + info.key + '" data-param="td-mean" type="number" min="0" max="10000" step="10" value="' + tdMeans[idx] + '" /></td>'
        ).join('') +
      '</tr>'
    );
    rows.push(
      '<tr>' +
        '<td class="sticky-col"></td>' +
        '<td>m</td>' +
        '<td>spread σ</td>' +
        unique.map((info, idx) =>
          '<td><input class="flight-config-input" data-ac="' + info.key + '" data-param="td-sigma" type="number" min="0" max="10000" step="10" value="' + tdSigmas[idx] + '" /></td>'
        ).join('') +
      '</tr>'
    );
    rows.push(
      '<tr>' +
        '<td class="sticky-col">Touchdown speed VTD</td>' +
        '<td>m/s</td>' +
        '<td>mean μ</td>' +
        unique.map((info, idx) =>
          '<td><input class="flight-config-input" data-ac="' + info.key + '" data-param="vtd-mean" type="number" min="0" max="150" step="1" value="' + vtdMeans[idx] + '" /></td>'
        ).join('') +
      '</tr>'
    );
    rows.push(
      '<tr>' +
        '<td class="sticky-col"></td>' +
        '<td>m/s</td>' +
        '<td>spread σ</td>' +
        unique.map((info, idx) =>
          '<td><input class="flight-config-input" data-ac="' + info.key + '" data-param="vtd-sigma" type="number" min="0" max="150" step="1" value="' + vtdSigmas[idx] + '" /></td>'
        ).join('') +
      '</tr>'
    );
    rows.push(
      '<tr>' +
        '<td class="sticky-col">Deceleration a</td>' +
        '<td>m/s²</td>' +
        '<td>mean μ</td>' +
        unique.map((info, idx) =>
          '<td><input class="flight-config-input" data-ac="' + info.key + '" data-param="a-mean" type="number" min="0" max="10" step="0.1" value="' + aMeans[idx] + '" /></td>'
        ).join('') +
      '</tr>'
    );
    rows.push(
      '<tr>' +
        '<td class="sticky-col"></td>' +
        '<td>m/s²</td>' +
        '<td>spread σ</td>' +
        unique.map((info, idx) =>
          '<td><input class="flight-config-input" data-ac="' + info.key + '" data-param="a-sigma" type="number" min="0" max="10" step="0.1" value="' + aSigmas[idx] + '" /></td>'
        ).join('') +
      '</tr>'
    );
    rows.push(
      '<tr>' +
        '<td class="sticky-col" style="background:rgba(124,106,247,0.14);">Distance to 26 m/s (from threshold)</td>' +
        '<td style="background:rgba(124,106,247,0.14);">m</td>' +
        '<td style="background:rgba(124,106,247,0.14);">mean-based</td>' +
        unique.map((info, idx) =>
          '<td style="background:rgba(124,106,247,0.14);font-weight:600;color:#ede9fe;">' + aMeanStopDists[idx] + '</td>'
        ).join('') +
      '</tr>'
    );
    const retStats = (typeof getScheduleRetStatsAll === 'function')
      ? getScheduleRetStatsAll()
      : (typeof computeRunwayExitDistances === 'function' ? computeRunwayExitDistances() : []);
    if (retStats && retStats.length) {
      rows.push(
        '<tr>' +
          '<td class="sticky-col" style="padding-top:10px;">Runway exits (distance from threshold)</td>' +
          '<td></td>' +
          '<td></td>' +
          unique.map(() => '<td></td>').join('') +
        '</tr>'
      );
      const byRunway = new Map();
      retStats.forEach(r => {
        const rwId = r && r.runway && r.runway.id ? String(r.runway.id) : '';
        const key = rwId || '__unknown__';
        if (!byRunway.has(key)) byRunway.set(key, []);
        byRunway.get(key).push(r);
      });
      function runwayGroupSortKey(rwKey) {
        if (!rwKey || rwKey === '__unknown__') return 'zzzz__unknown__';
        const disp = (typeof getRunwayDisplayLabelById === 'function') ? getRunwayDisplayLabelById(rwKey) : rwKey;
        return String(disp || rwKey);
      }
      const runwayKeysSorted = Array.from(byRunway.keys()).sort((a, b) => runwayGroupSortKey(a).localeCompare(runwayGroupSortKey(b)));
      runwayKeysSorted.forEach((rwKey, rwIdx) => {
        const list = byRunway.get(rwKey) || [];
        const rwLabel = (rwKey && rwKey !== '__unknown__')
          ? escapeHtml(getRunwayDisplayLabelById(rwKey) || rwKey)
          : '—';
        list
          .slice()
          .sort((a, b) => (a && isFinite(a.distM) ? a.distM : 0) - (b && isFinite(b.distM) ? b.distM : 0))
          .forEach((r, idxInRw) => {
            void idxInRw;
            const counts = unique.map(info => {
              const typeKey = info.key;
              return (state.flights || []).filter(f =>
                f.sampledArrRet === (r.exit && r.exit.id) &&
                arrivalConfigColumnKeyForFlight(f) === typeKey
              ).length;
            });
            const sortedIdx = counts
              .map((c, i) => [c, i])
              .filter(([c]) => c > 0)
              .sort((a, b) => b[0] - a[0]);
            const top1 = sortedIdx[0] ? sortedIdx[0][1] : -1;
            const top2 = sortedIdx[1] ? sortedIdx[1][1] : -1;
            const top3 = sortedIdx[2] ? sortedIdx[2][1] : -1;
            rows.push(
              '<tr>' +
                '<td class="sticky-col">' +
                  '<span style="display:inline-flex;align-items:center;gap:4px;">' +
                    '<span style="font-size:9px;color:#9ca3af;font-weight:700;">' + rwLabel + '</span>' +
                    '<span style="padding:2px 6px;border-radius:9999px;background:rgba(124,106,247,0.16);border:1px solid rgba(124,106,247,0.35);font-size:10px;color:#ede9fe;font-weight:600;">' +
                      escapeHtml(r.name) +
                    '</span>' +
                  '</span>' +
                '</td>' +
                '<td>m</td>' +
                '<td>' + Math.round(r.distM) + '</td>' +
                unique.map((info, colIdx) => {
                  const cnt = counts[colIdx] || 0;
                  if (!cnt) return '<td></td>';
                  let bg = 'rgba(39,29,61,0.72)';
                  let color = '#ede9fe';
                  if (colIdx === top1) {
                    bg = 'rgba(124,106,247,0.36)';
                    color = '#f5f3ff';
                  } else if (colIdx === top2 || colIdx === top3) {
                    bg = 'rgba(124,106,247,0.22)';
                    color = '#ede9fe';
                  }
                  return '<td style="background:' + bg + ';color:' + color + ';font-weight:600;text-align:center;">' + cnt + '</td>';
                }).join('') +
              '</tr>'
            );
          });
        const isLastGroup = rwIdx === runwayKeysSorted.length - 1;
        if (!isLastGroup) {
          rows.push(
            '<tr>' +
              '<td class="sticky-col" style="padding:6px 0;border-bottom:1px solid rgba(107,114,128,0.35);"></td>' +
              '<td style="padding:6px 0;border-bottom:1px solid rgba(107,114,128,0.35);"></td>' +
              '<td style="padding:6px 0;border-bottom:1px solid rgba(107,114,128,0.35);"></td>' +
              unique.map(() => '<td style="padding:6px 0;border-bottom:1px solid rgba(107,114,128,0.35);"></td>').join('') +
            '</tr>'
          );
        }
      });
      const failedCounts = unique.map(info => {
        const typeKey = info.key;
        return (state.flights || []).filter(f =>
          isFlightArrRetFailedInConfigTable(f, retStats) &&
          arrivalConfigColumnKeyForFlight(f) === typeKey
        ).length;
      });
      if (failedCounts.some(c => c > 0)) {
        const sortedFailed = failedCounts
          .map((c, i) => [c, i])
          .filter(([c]) => c > 0)
          .sort((a, b) => b[0] - a[0]);
        const fTop1 = sortedFailed[0] ? sortedFailed[0][1] : -1;
        const fTop2 = sortedFailed[1] ? sortedFailed[1][1] : -1;
        const fTop3 = sortedFailed[2] ? sortedFailed[2][1] : -1;
        rows.push(
          '<tr>' +
            '<td class="sticky-col">' +
              '<span style="padding:2px 6px;border-radius:9999px;background:rgba(127,29,29,0.9);border:1px solid #b91c1c;font-size:10px;color:#fee2e2;font-weight:600;">Failed</span>' +
            '</td>' +
            '<td></td>' +
            '<td></td>' +
            unique.map((info, colIdx) => {
              const cnt = failedCounts[colIdx] || 0;
              if (!cnt) return '<td></td>';
              let bg = 'rgba(30,30,30,0.9)';
              let color = '#fecaca';
              if (colIdx === fTop1) {
                bg = 'rgba(220,38,38,0.65)';
                color = '#fee2e2';
              } else if (colIdx === fTop2 || colIdx === fTop3) {
                bg = 'rgba(239,68,68,0.45)';
                color = '#fee2e2';
              }
              return '<td style="background:' + bg + ';color:' + color + ';font-weight:600;text-align:center;">' + cnt + '</td>';
            }).join('') +
          '</tr>'
        );
      }
    }
    function _fmtFlightPhysVal(v) {
      if (v == null || v === '') return '—';
      const n = Number(v);
      if (!isFinite(n)) return '—';
      const r = Math.round(n * 100) / 100;
      return (Math.abs(r - Math.round(r)) < 0.005) ? String(Math.round(r)) : String(r);
    }
    const perFlightBody = flightsSorted.map(function(f) {
      const ac = typeof getAircraftInfoByType === 'function' ? getAircraftInfoByType(f.aircraftType) : null;
      const typeLabel = ac ? (ac.name || ac.id || f.aircraftType || '—') : (f.aircraftType || '—');
      const arrRetFailed = isFlightCountedInArrivalConfigFailedRow(f, retStats);
      let retDisp = '—';
      if (arrRetFailed) retDisp = 'Failed';
      else if (f.sampledArrRet != null && retStats && retStats.length) {
        const retInfo = retStats.find(r => r.exit && r.exit.id === f.sampledArrRet);
        retDisp = retInfo ? (retInfo.name || 'RET') : 'RET';
      }
      const retCellInner = arrRetFailed ? 'Failed' : escapeHtml(retDisp);
      return '' +
        '<tr>' +
          '<td>' + escapeHtml(f.reg || '—') + '</td>' +
          '<td>' + escapeHtml(f.airlineCode || '—') + '</td>' +
          '<td>' + escapeHtml(f.flightNumber || '—') + '</td>' +
          '<td>' + escapeHtml(String(typeLabel)) + '</td>' +
          '<td style="text-align:right;font-variant-numeric:tabular-nums;">' + _fmtFlightPhysVal(f.arrVTdMs) + '</td>' +
          '<td style="text-align:right;font-variant-numeric:tabular-nums;">' + _fmtFlightPhysVal(f.arrDecelMs2) + '</td>' +
          '<td class="flight-td-arr-ret' + (arrRetFailed ? ' flight-td-arr-ret-failed' : '') + '" style="text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums;">' + retCellInner + '</td>' +
        '</tr>';
    }).join('');
    const perFlightBlock = '' +
      '<div class="flight-config-sampled-caption">' +
        '<span class="flight-config-sampled-caption-ko">항공기별 적용값 · 샘플링된 접지속도(VTD)와 활주로 감속도</span>' +
        '<span class="flight-config-sampled-caption-en">Per flight: sampled VTD &amp; deceleration (used after page reload / path compute)</span>' +
      '</div>' +
      '<div class="flight-config-sampled-scroll">' +
        '<table class="flight-schedule-table flight-config-per-flight-table">' +
          '<thead><tr>' +
            '<th>Reg</th>' +
            '<th>Airline</th>' +
            '<th>Flight</th>' +
            '<th>Aircraft type</th>' +
            '<th style="text-align:right;">VTD (m/s)</th>' +
            '<th style="text-align:right;">Decel (m/s²)</th>' +
            '<th style="text-align:right;">Arr RET</th>' +
          '</tr></thead>' +
          '<tbody>' + perFlightBody + '</tbody>' +
        '</table>' +
      '</div>';
    cfgEl.innerHTML = cfgHeader + rows.join('') + '</tbody></table>' +
      '<div style="font-size:10px;color:#6b7280;margin-top:8px;">' +
        'Note: sampling is clipped to stay within ±15% of each mean value.' +
      '</div>' +
      perFlightBlock;
  }

  function syncAllocGanttSelectionHighlight() {
    const ganttRoot = document.getElementById('allocationGantt');
    if (!ganttRoot || !ganttRoot.querySelector('.alloc-gantt-root')) return;
    ganttRoot.querySelectorAll('.alloc-flight').forEach(function(el) {
      el.classList.remove('alloc-flight-selected');
    });
    const sel = state.selectedObject;
    if (!sel || sel.type !== 'flight' || !sel.id) return;
    const wantId = String(sel.id);
    ganttRoot.querySelectorAll('.alloc-flight').forEach(function(el) {
      if (el.getAttribute('data-flight-id') === wantId) el.classList.add('alloc-flight-selected');
    });
  }

  function _flightListWireEvents(listEl, st) {
    listEl.querySelectorAll('.obj-item-delete').forEach(function(btn) {
      btn.addEventListener('click', function(ev) {
        var idVal = this.getAttribute('data-del');
        var fDel = st.flights.find(function(x) { return x.id === idVal; });
        var delStand = (fDel && fDel.standId) ? fDel.standId : null;
        st.flights = st.flights.filter(function(f) { return f.id !== idVal; });
        recomputeSimDuration();
        if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
        if (delStand)
          renderFlightList(false, false, { scheduleMode: 'incremental', dirtyFlightIds: [], touchedStandIds: [delStand] });
        else
          renderFlightList();
      });
    });
    listEl.querySelectorAll('.obj-item').forEach(function(row) {
      row.addEventListener('click', function(ev) {
        if ((ev.target.classList && ev.target.classList.contains('obj-item-delete')) || ev.target.getAttribute('data-del')) return;
        var idVal = this.getAttribute('data-id');
        var f = st.flights.find(function(x) { return x.id === idVal; });
        if (!f) return;
        state.flightPathRevealFlightId = null;
        st.selectedObject = { type: 'flight', id: idVal, obj: f };
        listEl.querySelectorAll('.obj-item').forEach(function(r) { r.classList.remove('selected', 'expanded'); });
        this.classList.add('selected', 'expanded');
        if (typeof updateObjectInfo === 'function') updateObjectInfo();
        if (typeof syncPanelFromState === 'function') syncPanelFromState();
        if (typeof draw === 'function') draw();
        if (typeof syncAllocGanttSelectionHighlight === 'function') syncAllocGanttSelectionHighlight();
      });
      row.addEventListener('dblclick', function(ev) {
        if ((ev.target.classList && ev.target.classList.contains('obj-item-delete')) || ev.target.getAttribute('data-del')) return;
        ev.preventDefault();
        var idVal = this.getAttribute('data-id');
        var f = st.flights.find(function(x) { return x.id === idVal; });
        if (!f) return;
        st.selectedObject = { type: 'flight', id: idVal, obj: f };
        state.flightPathRevealFlightId = idVal;
        listEl.querySelectorAll('.obj-item').forEach(function(r) { r.classList.remove('selected', 'expanded'); });
        this.classList.add('selected', 'expanded');
        if (typeof updateObjectInfo === 'function') updateObjectInfo();
        if (typeof syncPanelFromState === 'function') syncPanelFromState();
        if (typeof draw === 'function') draw();
        if (typeof syncAllocGanttSelectionHighlight === 'function') syncAllocGanttSelectionHighlight();
      });
    });
  }


  function _ganttSaveViewState(ganttEl) {
    let scrollLeft = 0, scrollTop = 0;
    const scrollCol = ganttEl.querySelector('.alloc-gantt-scroll-col');
    if (scrollCol) {
      scrollLeft = scrollCol.scrollLeft || 0;
      scrollTop = scrollCol.scrollTop || 0;
    }
    const collapsedTerminals = new Set();
    let remoteCollapsed = false;
    const labelCol = ganttEl.querySelector('.alloc-gantt-label-col');
    if (labelCol) {
      Array.from(labelCol.children).forEach(function (el) {
        if (el.classList && el.classList.contains('alloc-terminal-header')) {
          if (el.getAttribute('data-collapsed') === '1') {
            let txt = (el.textContent || '').trim().replace(/^[▶▼]\s*/, '');
            if (txt) collapsedTerminals.add(txt);
          }
        }
        if (el.classList && el.classList.contains('alloc-remote-header')) {
          if (el.getAttribute('data-collapsed') === '1') remoteCollapsed = true;
        }
      });
    }
    return { scrollLeft: scrollLeft, scrollTop: scrollTop, collapsedTerminals: collapsedTerminals, remoteCollapsed: remoteCollapsed };
  }

  function renderFlightGantt(opt) {
    const skipPathPrep = opt && opt.skipPathPrep;
    const ganttEl = document.getElementById('allocationGantt');
    if (!ganttEl) return;
    const viewState = _ganttSaveViewState(ganttEl);
    const prevScrollLeft = viewState.scrollLeft;
    const prevScrollTop = viewState.scrollTop;
    const prevCollapsedTerminals = viewState.collapsedTerminals;
    const prevRemoteCollapsed = viewState.remoteCollapsed;
    if (!state.flights.length) {
      state.allocGanttWindowStartMin = null;
      ganttEl.innerHTML = '<div style="font-size:11px;color:#9ca3af;">No flights for Gantt.</div>';
      const labEmpty = document.getElementById('allocGanttWindowLabel');
      if (labEmpty) labEmpty.textContent = '';
      return;
    }
    const flights = state.flights.slice();
    const stands = allStandsForFlightAssignment();
    if (!flights.length) {
      state.allocGanttWindowStartMin = null;
      ganttEl.innerHTML = '<div style="font-size:11px;color:#9ca3af;">No flights for Gantt.</div>';
      const labEmpty2 = document.getElementById('allocGanttWindowLabel');
      if (labEmpty2) labEmpty2.textContent = '';
      return;
    }
    if (!skipPathPrep) {
      if (typeof computeScheduledDisplayTimes === 'function') computeScheduledDisplayTimes(state.flights);
    }

    let intervals = [];
    const schedTable = document.querySelector('.flight-schedule-table');
    const domScheduleOk = schedTable && schedTable.getAttribute('data-virtual-table') !== '1';
    if (domScheduleOk) {
      const rows = Array.from(schedTable.querySelectorAll('tbody tr.flight-data-row'));
      const flightById = new Map();
      for (let fi = 0; fi < flights.length; fi++) {
        const ff = flights[fi];
        if (ff && ff.id != null) flightById.set(String(ff.id), ff);
      }
      rows.forEach(row => {
        const id = row.getAttribute('data-id');
        if (!id) return;
        const f = flightById.get(String(id));
        if (!f) return;
        const tds = Array.from(row.querySelectorAll('td'));
        if (tds.length <= FLIGHT_SCHED_TD_ETOT) return;
        const getMin = (idx) => {
          const td = tds[idx];
          if (!td) return 0;
          const dm = td.getAttribute('data-sched-min');
          if (dm != null && String(dm).trim() !== '') {
            const n = parseFloat(dm);
            return isFinite(n) ? n : 0;
          }
          const txt = (td.textContent || '').trim();
          if (!txt) return 0;
          try {
            return parseTimeToMinutes(txt);
          } catch (e) {
            return 0;
          }
        };
        const sibt_d = getMin(FLIGHT_SCHED_TD_SIBTD);
        const sobt_d = getMin(FLIGHT_SCHED_TD_SOBTD);
        const sldt_d = Math.max(0, sibt_d - SCHED_SD_SIBT_MINUS_SLD_MIN);
        const stot_d = sobt_d + SCHED_SD_STOT_PLUS_SOBD_MIN;
        const eSer = ganttESeriesMinutesFromTimelineMeta(f);
        const eldt = eSer.eldt;
        const eibt = eSer.eibt;
        const eobt = eSer.eobt;
        const etot = eSer.etot;
        const t0 = sibt_d;
        const t1 = sobt_d || (t0 + (f.dwellMin != null ? f.dwellMin : 0));
        const sldt = sldt_d || t0;
        const stot = stot_d || t1;
        const sldtOrig = sldt;
        const sobtOrig = sobt_d || t1;
        const stotOrig = stot;
        intervals.push({ f, t0, t1, sldt, stot, eibt, eobt, eldt, etot, sldtOrig, sobtOrig, stotOrig });
      });
    }
    if (!intervals.length) {
      intervals = flights.map(f => {
        const t0 = f.sibtMin_d != null ? f.sibtMin_d : (f.timeMin != null ? f.timeMin : 0);
        const t1 = f.sobtMin_d != null ? f.sobtMin_d : (t0 + (f.dwellMin != null ? f.dwellMin : 0));
        const sldt = f.sldtMin_d != null ? f.sldtMin_d : t0;
        const stot = f.stotMin_d != null ? f.stotMin_d : t1;
        const eSer2 = ganttESeriesMinutesFromTimelineMeta(f);
        const eibt = eSer2.eibt;
        const eobt = eSer2.eobt;
        const eldt = eSer2.eldt;
        const etot = eSer2.etot;
        const sldtOrig = sldt;
        const sobtOrig = f.sobtMin_d != null ? f.sobtMin_d : t1;
        const stotOrig = stot;
        return { f, t0, t1, sldt, stot, eibt, eobt, eldt, etot, sldtOrig, sobtOrig, stotOrig };
      });
    }

    let minS = Infinity;
    let maxE = -Infinity;
    intervals.forEach(it => {
      if (it.sldt < minS) minS = it.sldt;
      const etot0 = (it.etot != null && isFinite(it.etot)) ? it.etot : it.stot;
      if (etot0 > maxE) maxE = etot0;
    });
    if (minS <= 0 && intervals.length) {
      const posSldt = intervals.map(function(it) { return it.sldt; }).filter(function(v) { return isFinite(v) && v > 1e-6; });
      if (posSldt.length) minS = Math.min.apply(null, posSldt);
    }
    if (!isFinite(minS) || !isFinite(maxE)) {
      ganttEl.innerHTML = '';
      return;
    }
    const baseMinT = Math.max(0, minS - GANTT_PAD_MIN);
    const baseMaxT0 = maxE + GANTT_PAD_MIN;
    const baseMaxT = (baseMaxT0 <= baseMinT) ? (baseMinT + 60) : baseMaxT0;
    const baseSpan = baseMaxT - baseMinT;
    const dataSpan = Math.max(1e-9, baseSpan);
    const visibleSpan = Math.min(GANTT_VISIBLE_WINDOW_MIN, dataSpan);
    let winStart = state.allocGanttWindowStartMin;
    if (winStart == null || !isFinite(winStart)) winStart = baseMinT;
    const maxWinStart = Math.max(baseMinT, baseMaxT - visibleSpan);
    winStart = Math.min(Math.max(winStart, baseMinT), maxWinStart);
    state.allocGanttWindowStartMin = winStart;
    const winEnd = winStart + visibleSpan;
    state._allocGanttClamp = { baseMinT: baseMinT, baseMaxT: baseMaxT, visibleSpan: visibleSpan };
    const displaySpan = visibleSpan;
    const zoom = (state.allocTimeZoom && state.allocTimeZoom > 1) ? state.allocTimeZoom : 1;

    const tickPositions = buildTimeAxisTicks(winStart, winEnd, winStart, displaySpan, zoom);

    function allocLeftPct(t) {
      return ((t - winStart) / displaySpan) * 100 * zoom;
    }
    function allocTrackSpanHtml(cls, leftPct, widthPct, minWidthPct) {
      return '<div class="' + cls + '" style="left:' + leftPct + '%;width:' + Math.max(minWidthPct, widthPct) + '%;"></div>';
    }
    function allocTrackMarkerHtml(cls, leftPct) {
      return '<div class="' + cls + '" style="left:' + leftPct + '%;"></div>';
    }
    function pushAllocDot(arr, t, cls) {
      if (!arr || !isFinite(t) || t < winStart || t > winEnd) return;
      arr.push(allocTrackMarkerHtml('alloc-time-dot ' + cls, allocLeftPct(t)));
    }
    function pushAllocSpan(arr, startT, endT, cls, minWidthPct) {
      if (!arr || !isFinite(startT) || !isFinite(endT) || endT <= startT) return;
      const clippedStart = Math.max(startT, winStart);
      const clippedEnd = Math.min(endT, winEnd);
      if (clippedEnd <= clippedStart) return;
      arr.push(allocTrackSpanHtml(cls, allocLeftPct(clippedStart), ((clippedEnd - clippedStart) / displaySpan) * 100 * zoom, minWidthPct));
    }
    function pushAllocTriangle(arr, t, cls) {
      if (!arr || !isFinite(t) || t < winStart || t > winEnd) return;
      arr.push(allocTrackMarkerHtml(cls, allocLeftPct(t)));
    }

    /** O(flights) — avoid per-row intervals.filter (was O(stands * flights) per gantt pass). */
    const intervalsByStandKey = (function() {
      const o = { __unassigned: [] };
      for (let gi = 0; gi < intervals.length; gi++) {
        const it = intervals[gi];
        const raw = it.f && it.f.standId;
        if (raw == null || raw === '') o.__unassigned.push(it);
        else {
          const sid = String(raw);
          if (!o[sid]) o[sid] = [];
          o[sid].push(it);
        }
      }
      return o;
    })();

    function buildRowHtml(label, standId) {
      const showSPointsEl = document.getElementById('chkShowSPoints');
      const showSPoints = !showSPointsEl || showSPointsEl.checked;
      const showSBarsEl = document.getElementById('chkShowSBars');
      const dimSBars = !!(showSBarsEl && !showSBarsEl.checked);
      const showEBarEl = document.getElementById('chkShowEBar');
      const showEBar = !showEBarEl || showEBarEl.checked;
      const showEPointsEl = document.getElementById('chkShowEPoints');
      const showEPoints = !showEPointsEl || showEPointsEl.checked;
      const showAuxBars = showSPoints;
      const showEibtBars = showEBar;
      const showEldtBars = showEPoints;
      const showSDots = showSPoints;
      const showSdDots = showSPoints;
      const showEDots = showEPoints;
      const rowFlights = (standId == null)
        ? (intervalsByStandKey.__unassigned || [])
        : (intervalsByStandKey[String(standId)] || []);
      const conflictMap = {};
      for (let i = 0; i < rowFlights.length; i++) {
        for (let j = i + 1; j < rowFlights.length; j++) {
          const a = rowFlights[i];
          const b = rowFlights[j];
          if (a.t0 < b.t1 && b.t0 < a.t1) { // Section overlap
            conflictMap[a.f.id] = true;
            conflictMap[b.f.id] = true;
          }
        }
      }
      const sBars = showAuxBars ? [] : null;
      const eBars = showEibtBars ? [] : null;
      const e2Bars = showEldtBars ? [] : null;
      const sDots = showSDots ? [] : null;
      const sdDots = showSdDots ? [] : null;
      const eDots = showEDots ? [] : null;
      const sLines = showSPoints ? [] : null;      // SOBT(orig) vertical line
      const sTrisDown = showSPoints ? [] : null;   // SLDTtriangle under dragon
      const sTrisUp = showSPoints ? [] : null;     // STOTtriangle above dragon
      const eTrisDown = showEPoints ? [] : null;   // ELDTtriangle under dragon
      const eTrisUp = showEPoints ? [] : null;     // ETOTtriangle above dragon
      const blocks = rowFlights.map(it => {
        const f = it.f;
        const t0 = it.t0;
        const t1 = it.t1;
        const sldt = it.sldt;
        const stot = it.stot;
        const eibt = it.eibt;
        const eobt = it.eobt;
        const eldt = it.eldt;
        const etot = it.etot;
        const depBlk = (typeof getDepBlockOutMin === 'function') ? getDepBlockOutMin(f) : 0;
        const sobtOrig = (it.sobtOrig != null) ? it.sobtOrig : (it.stotOrig - depBlk);
        const tStart = Math.max(t0, winStart);
        const tEnd = Math.min(t1, winEnd);
        if (tEnd <= tStart) return '';
        const leftPct = ((tStart - winStart) / displaySpan) * 100 * zoom;
        const widthPct = Math.max(2, ((tEnd - tStart) / displaySpan) * 100 * zoom);
        const regSafe = escapeHtml(f.reg || '');
        const codeSafe = escapeHtml((f.code || '').toUpperCase());
        const typeSafe = escapeHtml(String(f.aircraftType || '').trim());
        const codeHtml = codeSafe ? ('<span class="alloc-flight-code">' + codeSafe + '</span>') : '';
        const typeHtml = typeSafe
          ? ((codeSafe ? '<span class="alloc-flight-type-sep"> · </span>' : '') + '<span class="alloc-flight-type">' + typeSafe + '</span>')
          : '';
        const metaHtml = (codeHtml || typeHtml)
          ? ('<div class="alloc-flight-meta">' + codeHtml + typeHtml + '</div>')
          : '';
        const conflictClass = (conflictMap[f.id] || flightBlockedLikeNoWay(f)) ? ' conflict' : '';
        const selectedClass = (state.selectedObject && state.selectedObject.type === 'flight' && state.selectedObject.id === f.id) ? ' alloc-flight-selected' : '';
        const sbarDimClass = dimSBars ? ' alloc-flight-sbar-dim' : '';
        const sibtLabel = formatFlightScheduleDateTime(f, t0);
        const sobtLabel = formatFlightScheduleDateTime(f, t1);
        const barTitle =
          'SIBT: ' + sibtLabel +
          '\\nSOBT: ' + sobtLabel +
          '\\nReg: ' + (f.reg || '') +
          '\\nAirline: ' + (f.airlineCode || '') + ' ' + (f.flightNumber || '');
        if (showEibtBars && eBars && isFinite(eibt) && isFinite(eobt) && eobt > eibt) {
          pushAllocSpan(eBars, eibt, eobt, 'alloc-e-bar', 2);
        }
        const hasOverlap = (f.vttADelayMin != null && f.vttADelayMin > 0) || f.eOverlapPushed;
        const ovlpBadgeHtml = hasOverlap ? '<span class="alloc-flight-ovlp-badge">OVLP</span>' : '';
        if (showEldtBars && e2Bars) {
          if (isFinite(eldt) && isFinite(eibt) && eibt >= eldt) pushAllocSpan(e2Bars, eldt, eibt, 'alloc-e2-bar', 0.5);
          if (isFinite(eobt) && isFinite(etot) && etot >= eobt) pushAllocSpan(e2Bars, eobt, etot, 'alloc-e2-bar', 0.5);
        }
        if (showAuxBars && sBars) {
          if (isFinite(sldt) && sldt <= t0) pushAllocSpan(sBars, sldt, t0, 'alloc-s-bar', 0.5);
          if (isFinite(stot) && stot >= t1) pushAllocSpan(sBars, t1, stot, 'alloc-s-bar', 0.5);
        }
        if (showSDots && sDots) {
          pushAllocDot(sDots, sldt, 'alloc-time-dot-s');
          pushAllocDot(sDots, stot, 'alloc-time-dot-s');
        }
        if (showSdDots && sdDots) {
          pushAllocDot(sdDots, sldt, 'alloc-time-dot-sd');
          pushAllocDot(sdDots, stot, 'alloc-time-dot-sd');
        }
        if (showEDots && eDots) {
          pushAllocDot(eDots, eldt, 'alloc-time-dot-e');
          pushAllocDot(eDots, etot, 'alloc-time-dot-e');
          pushAllocTriangle(eTrisDown, eldt, 'alloc-e-tri alloc-e-tri-down');
          pushAllocTriangle(eTrisUp, etot, 'alloc-e-tri alloc-e-tri-up');
        }
        if (showSPoints) {
          pushAllocTriangle(sTrisDown, sldt, 'alloc-s-tri alloc-s-tri-down');
          pushAllocTriangle(sTrisUp, stot, 'alloc-s-tri alloc-s-tri-up');
        }
      if (sLines && ((f.vttADelayMin != null && f.vttADelayMin > 0) || f.eOverlapPushed) && isFinite(sobtOrig)) {
        const sobtD = (f.sobtMin_d != null ? f.sobtMin_d : t1);
        if (!isNaN(sobtD) && Math.abs(sobtOrig - sobtD) > 1e-6) {
          const sx = ((sobtOrig - winStart) / displaySpan) * 100 * zoom;
          sLines.push('<div class="alloc-s-line-orig" style="left:' + sx + '%;"></div>');
        }
      }
        return '' +
          '<div class="alloc-flight' + conflictClass + selectedClass + sbarDimClass + '" draggable="true" data-flight-id="' + f.id + '" ' +
            'style="left:' + leftPct + '%;width:' + widthPct + '%;min-width:4px;"' +
            ' title="' + barTitle + '">' +
            '<div class="alloc-flight-reg">' + regSafe + '</div>' +
            metaHtml +
            ovlpBadgeHtml +
          '</div>';
      }).join('');
      const sidAttr = standId ? String(standId) : '';
      const apronDropOk = standId == null || standHasApronTaxiwayLink(standId);
      const rowNoLinkClass = (!apronDropOk && standId != null) ? ' alloc-row-no-apron-link' : '';
      const apronLinkDataAttr = ' data-apron-link-ok="' + (apronDropOk ? '1' : '0') + '"';
      const bgSlots = (tickPositions.length > 1)
        ? tickPositions.slice(0, -1).map((tp, idx) => {
            const next = tickPositions[idx + 1];
            const midLeft = (tp.leftPct + next.leftPct) / 2;
            return (
              '<div class="alloc-apron-bg-slot" style="left:' + midLeft + '%;transform:translateX(-50%);">' +
                escapeHtml(label) +
              '</div>'
            );
          }).join('')
        : '';
      const labelHtml =
        '<div class="alloc-row-label' + rowNoLinkClass + '" data-stand-id="' + sidAttr + '"' + apronLinkDataAttr + '>' +
          escapeHtml(label) +
        '</div>';
      const trackHtml =
        '<div class="alloc-row' + rowNoLinkClass + '" data-stand-id="' + sidAttr + '"' + apronLinkDataAttr + '>' +
          '<div class="alloc-row-track" data-stand-id="' + sidAttr + '"' + apronLinkDataAttr + '>' +
            bgSlots +
            blocks +
            (showEibtBars && eBars ? eBars.join('') : '') +
            (showEldtBars && e2Bars ? e2Bars.join('') : '') +
            (showAuxBars && sBars ? sBars.join('') : '') +
            (showSDots && sDots ? sDots.join('') : '') +
            (showSdDots && sdDots ? sdDots.join('') : '') +
            (showEDots && eDots ? eDots.join('') : '') +
            (sTrisDown ? sTrisDown.join('') : '') +
            (sTrisUp ? sTrisUp.join('') : '') +
            (eTrisDown ? eTrisDown.join('') : '') +
            (eTrisUp ? eTrisUp.join('') : '') +
            (sLines ? sLines.join('') : '') +
          '</div>' +
        '</div>';
      return { labelHtml, trackHtml };
    }
    function buildRunwayLegendPair() {
      const sDotsHtml = [];
      const eDotsHtml = [];
      const cap = GANTT_LEGEND_MAX_INTERVALS;
      const lim = (cap > 0 && intervals.length > cap) ? intervals.slice(0, cap) : intervals;
      lim.forEach(function(it) {
        pushAllocDot(sDotsHtml, it.sldt, 'alloc-time-dot-s');
        pushAllocDot(sDotsHtml, it.stot, 'alloc-time-dot-s');
        pushAllocDot(eDotsHtml, it.eldt, 'alloc-time-dot-e');
        pushAllocDot(eDotsHtml, it.etot, 'alloc-time-dot-e');
      });
      const sLabelHtml = '<div class="alloc-row-label alloc-runway-legend-label" data-stand-id="" data-runway-legend="1">' + escapeHtml('S(LDT, TOT)') + '</div>';
      const sTrackHtml =
        '<div class="alloc-row" data-stand-id="" data-runway-legend="1">' +
          '<div class="alloc-row-track" data-stand-id="" data-runway-legend="1" style="background:transparent;border:none;">' +
            sDotsHtml.join('') +
          '</div>' +
        '</div>';
      const eLabelHtml = '<div class="alloc-row-label alloc-runway-legend-label" data-stand-id="" data-runway-legend="1">' + escapeHtml('E(LDT, TOT)') + '</div>';
      const eTrackHtml =
        '<div class="alloc-row" data-stand-id="" data-runway-legend="1">' +
          '<div class="alloc-row-track" data-stand-id="" data-runway-legend="1" style="background:transparent;border:none;">' +
            eDotsHtml.join('') +
          '</div>' +
        '</div>';
      return { sLabelHtml: sLabelHtml, sTrackHtml: sTrackHtml, eLabelHtml: eLabelHtml, eTrackHtml: eTrackHtml };
    }
    const labelRows = [];
    const trackRows = [];
    (function() {
      const rw = buildRunwayLegendPair();
      labelRows.push(rw.sLabelHtml);
      trackRows.push(rw.sTrackHtml);
      labelRows.push(rw.eLabelHtml);
      trackRows.push(rw.eTrackHtml);
    })();
    (function() {
      const row = buildRowHtml('Unassigned', null);
      labelRows.push(row.labelHtml);
      trackRows.push(row.trackHtml);
    })();
    const terminalCopies = makeUniqueNamedCopy(state.terminals || [], 'name');
    const termLabelById = {};
    terminalCopies.forEach(t => { termLabelById[t.id] = (t.name || '').trim() || 'Building'; });
    const terminalIdsWithApronLink = (function() {
      const s = new Set();
      const links = state.apronLinks || [];
      for (let i = 0; i < links.length; i++) {
        const lk = links[i];
        if (!lk || !lk.pbbId) continue;
        const pbb = (state.pbbStands || []).find(function(p) { return p && p.id === lk.pbbId; });
        const rem = (state.remoteStands || []).find(function(r) { return r && r.id === lk.pbbId; });
        const tmp = (state.tempStands || []).find(function(r) { return r && r.id === lk.pbbId; });
        const st = pbb || rem || tmp;
        if (!st) continue;
        const t = getTerminalForStand(st);
        if (t && t.id != null) s.add(String(t.id));
      }
      return s;
    })();
    const pbbStandIdSet = new Set((state.pbbStands || []).map(function(p) { return p && p.id; }).filter(Boolean));
    const ganttTermByStand = new Map();
    stands.forEach(function(s) {
      const term = getTerminalForStand(s);
      if (!term) {
        ganttTermByStand.set(s.id, null);
        return;
      }
      const hasPbb = pbbStandIdSet.has(s.id);
      ganttTermByStand.set(
        s.id,
        (hasPbb || terminalIdsWithApronLink.has(String(term.id))) ? term : null
      );
    });
    const grouped = {};
    const order = [];
    const sortedStands = stands.slice().sort((a, b) => {
      const ta = ganttTermByStand.get(a.id);
      const tb = ganttTermByStand.get(b.id);
      const la = ta ? (termLabelById[ta.id] || ta.name || '') : '';
      const lb = tb ? (termLabelById[tb.id] || tb.name || '') : '';
      if (la < lb) return -1;
      if (la > lb) return 1;
      const na = (a.name || '').toLowerCase();
      const nb = (b.name || '').toLowerCase();
      if (na < nb) return -1;
      if (na > nb) return 1;
      return 0;
    });
    sortedStands.forEach(s => {
      const term = ganttTermByStand.get(s.id);
      const key = term ? term.id : '__no_terminal__';
      if (!grouped[key]) {
        grouped[key] = { term, stands: [] };
        order.push(key);
      }
      grouped[key].stands.push(s);
    });
    const remoteIdSet = new Set(
      (state.remoteStands || []).map(r => r.id)
        .concat((state.tempStands || []).map(r => r.id))
    );
    const allRemoteStands = [];
    order.forEach(key => {
      const group = grouped[key];
      if (!group) return;
      const term = group.term;
      const contactStands = [];
      const remoteStandsInTerm = [];
      group.stands.forEach(s => {
        if (remoteIdSet.has(s.id)) remoteStandsInTerm.push(s);
        else contactStands.push(s);
      });
      if (remoteStandsInTerm.length) {
        remoteStandsInTerm.forEach(s => allRemoteStands.push(s));
      }
      if (!contactStands.length) return;
      const headerLabel = term
        ? (termLabelById[term.id] || term.name || 'Building')
        : 'No Building';
      const headerEsc = escapeHtml(headerLabel);
      labelRows.push(
        '<div class="alloc-terminal-header" data-collapsed="0" title="' + headerEsc + '">' +
          '<span class="alloc-section-toggle-icon">▼</span>' +
          '<span class="alloc-terminal-header-text">' + headerEsc + '</span>' +
        '</div>'
      );
      trackRows.push('<div class="alloc-row" data-stand-id="">' +
        '<div class="alloc-row-track" data-stand-id="" style="background:transparent;border:none;height:20px;"></div>' +
      '</div>');
      contactStands.forEach(s => {
        const label = (s.name || '') + ' (' + (s.category || '') + ')';
        const row = buildRowHtml(label, s.id);
        labelRows.push(row.labelHtml);
        trackRows.push(row.trackHtml);
      });
    });
    if (allRemoteStands.length) {
      labelRows.push('<div class="alloc-gantt-section-spacer" aria-hidden="true"></div>');
      trackRows.push(
        '<div class="alloc-row" data-stand-id="">' +
          '<div class="alloc-row-track" data-stand-id="" style="background:transparent;border:none;height:4px;min-height:4px;"></div>' +
        '</div>'
      );
      labelRows.push(
        '<div class="alloc-remote-header" data-collapsed="0">' +
          '<span class="alloc-section-toggle-icon">▼</span>' +
          'Remote stands' +
        '</div>'
      );
      trackRows.push(
        '<div class="alloc-row" data-stand-id="">' +
          '<div class="alloc-row-track" data-stand-id="" style="background:transparent;border:none;height:18px;min-height:18px;"></div>' +
        '</div>'
      );
      allRemoteStands.forEach(s => {
        const label = (s.name || '') + ' (' + (s.category || '') + ')';
        const row = buildRowHtml(label, s.id);
        labelRows.push(row.labelHtml);
        trackRows.push(row.trackHtml);
      });
    }
    const axisTicks = tickPositions.map(tp =>
      '<div class="alloc-time-tick" style="left:' + tp.leftPct + '%;">' +
        '<div class="alloc-time-tick-label">' + tp.label + '</div>' +
      '</div>'
    );
    const axisHtml =
      '<div class="alloc-time-axis-overlay">' +
        '<div class="alloc-time-axis-inner">' + axisTicks.join('') + '</div>' +
      '</div>';

    labelRows.push('<div class="alloc-label-axis-spacer"></div>');

    const labelColHtml =
      '<div class="alloc-gantt-label-col">' +
        labelRows.join('') +
      '</div>';
    const innerMinWidthPct = Math.max(100, Math.round(zoom * 100));
    const gridOverlayHtml =
      '<div class="alloc-gantt-grid-overlay">' +
        tickPositions.map(function(tp) {
          return '<div class="alloc-time-grid-line" style="left:' + tp.leftPct + '%;"></div>';
        }).join('') +
      '</div>';
    const trackColHtml =
      '<div class="alloc-gantt-scroll-col">' +
        '<div class="alloc-gantt-inner" style="min-width:' + innerMinWidthPct + '%;">' +
          gridOverlayHtml +
          trackRows.join('') +
          axisHtml +
        '</div>' +
      '</div>';
    const rootHtml =
      '<div class="alloc-gantt-root">' +
        labelColHtml +
        trackColHtml +
      '</div>';

    ganttEl.innerHTML = rootHtml;
    const labWin = document.getElementById('allocGanttWindowLabel');
    if (labWin) labWin.textContent = formatMinutesToHHMM(winStart) + ' – ' + formatMinutesToHHMM(winEnd);
    if (!state._allocGanttPanWired) {
      state._allocGanttPanWired = true;
      const bPrev = document.getElementById('btnAllocGanttPrev');
      const bNext = document.getElementById('btnAllocGanttNext');
      function allocGanttPanStep(deltaMin) {
        const c = state._allocGanttClamp;
        if (!c) return;
        let w = state.allocGanttWindowStartMin != null ? state.allocGanttWindowStartMin : c.baseMinT;
        w += deltaMin;
        const maxW = Math.max(c.baseMinT, c.baseMaxT - c.visibleSpan);
        state.allocGanttWindowStartMin = Math.min(Math.max(w, c.baseMinT), maxW);
        renderFlightGantt({ skipPathPrep: true });
      }
      if (bPrev) bPrev.addEventListener('click', function() { allocGanttPanStep(-GANTT_PAN_STEP_MIN); });
      if (bNext) bNext.addEventListener('click', function() { allocGanttPanStep(GANTT_PAN_STEP_MIN); });
    }
    const newScrollCol = ganttEl.querySelector('.alloc-gantt-scroll-col');
    const newLabelCol = ganttEl.querySelector('.alloc-gantt-label-col');
    if (newScrollCol) {
      if (prevScrollLeft > 0) newScrollCol.scrollLeft = prevScrollLeft;
      if (prevScrollTop > 0) newScrollCol.scrollTop = prevScrollTop;
    }
    if (newScrollCol && newLabelCol) {
      newScrollCol.addEventListener('scroll', function() { newLabelCol.scrollTop = newScrollCol.scrollTop; });
      newLabelCol.addEventListener('scroll', function() { newScrollCol.scrollTop = newLabelCol.scrollTop; });
    }
    if (newScrollCol && newLabelCol) {
      const labelChildren = Array.from(newLabelCol.children);
      const innerEl = newScrollCol.querySelector('.alloc-gantt-inner');
      const trackChildren = innerEl ? Array.from(innerEl.children).filter(function(el) {
        return el.classList.contains('alloc-row');
      }) : [];
      function _toggleSectionRows(labelArr, trackArr, fromIdx, collapsed) {
        const STOP = ['alloc-terminal-header','alloc-remote-header','alloc-label-axis-spacer','alloc-gantt-section-spacer'];
        for (let j = fromIdx; j < labelArr.length; j++) {
          const lbl = labelArr[j];
          if (STOP.some(function(c) { return lbl.classList.contains(c); })) break;
          lbl.style.display = collapsed ? 'none' : '';
          if (trackArr[j]) trackArr[j].style.display = collapsed ? 'none' : '';
        }
      }
      function _wireSectionHeader(el, idx, shouldStartCollapsed) {
        el.style.cursor = 'pointer';
        if (shouldStartCollapsed) {
          el.setAttribute('data-collapsed', '1');
          const icon0 = el.querySelector('.alloc-section-toggle-icon');
          if (icon0) icon0.textContent = '▶';
          _toggleSectionRows(labelChildren, trackChildren, idx + 1, true);
        }
        el.addEventListener('click', function() {
          const wasCollapsed = el.getAttribute('data-collapsed') === '1';
          const nowCollapsed = !wasCollapsed;
          el.setAttribute('data-collapsed', nowCollapsed ? '1' : '0');
          const icon = el.querySelector('.alloc-section-toggle-icon');
          if (icon) icon.textContent = nowCollapsed ? '▶' : '▼';
          _toggleSectionRows(labelChildren, trackChildren, idx + 1, nowCollapsed);
        });
      }
      labelChildren.forEach(function(el, idx) {
        if (el.classList.contains('alloc-terminal-header')) {
          let txt = (el.textContent || '').trim().replace(/^[▶▼]\s*/, '');
          _wireSectionHeader(el, idx, txt && prevCollapsedTerminals.has(txt));
        }
        if (el.classList.contains('alloc-remote-header')) {
          _wireSectionHeader(el, idx, prevRemoteCollapsed);
        }
      });
    }
    if (newScrollCol && !newScrollCol._allocWheelBound) {
      newScrollCol._allocWheelBound = true;
      newScrollCol.addEventListener('wheel', function(ev) {
        if (!ev.ctrlKey) return;
        ev.preventDefault();
        const delta = ev.deltaY || ev.deltaX || 0;
        newScrollCol.scrollLeft += delta;
      }, { passive: false });
    }

    _ganttWireInteractions(ganttEl, state);
  }

  function _ganttFindTrackAtPoint(scrollCol, clientX, clientY) {
    if (!scrollCol) return null;
    const inner = scrollCol.querySelector('.alloc-gantt-inner');
    if (!inner) return null;
    const rows = inner.querySelectorAll('.alloc-row');
    const tol = 2;
    for (let i = 0; i < rows.length; i++) {
      const r = rows[i].getBoundingClientRect();
      if (clientY >= r.top - tol && clientY <= r.bottom + tol) {
        const track = rows[i].querySelector('.alloc-row-track');
        if (track) return track;
      }
    }
    return null;
  }

  var _allocGanttPreviewTimer = null;
  var _allocGanttPreviewLastKey = '';
  function _allocGanttDragStandPreviewAllowed(f, standId) {
    if (!standId) return true;
    var allStands = allStandsForFlightAssignment();
    var stand = allStands.find(function(s) { return s.id === standId; });
    if (!stand) return false;
    return typeof flightCanUseStand === 'function' ? flightCanUseStand(f, stand) : true;
  }
  function _scheduleAllocGanttDragSchedulePreview(st, candStandId) {
    var ctxAtSchedule = st._allocGanttDrag;
    if (!ctxAtSchedule || !ctxAtSchedule.flightId) return;
    var seqWant = ctxAtSchedule.seq;
    if (_allocGanttPreviewTimer) clearTimeout(_allocGanttPreviewTimer);
    _allocGanttPreviewTimer = setTimeout(function() {
      _allocGanttPreviewTimer = null;
      var ctx = st._allocGanttDrag;
      if (!ctx || !ctx.flightId || ctx.seq !== seqWant) return;
      var f = st.flights.find(function(x) { return x.id === ctx.flightId; });
      if (!f) return;
      var sid = candStandId || null;
      if (!_allocGanttDragStandPreviewAllowed(f, sid)) return;
      var key = ctx.flightId + '|' + (sid || '');
      if (key === _allocGanttPreviewLastKey) return;
      _allocGanttPreviewLastKey = key;
      f.standId = sid;
      if (f.token) f.token.apronId = sid;
      var touched = [];
      if (ctx.prevStandId) touched.push(ctx.prevStandId);
      if (sid) touched.push(sid);
      if (typeof renderFlightList === 'function') {
        renderFlightList(false, false, { scheduleMode: 'incremental', dirtyFlightIds: [ctx.flightId], touchedStandIds: touched, skipGanttRefresh: true });
      }
      if (typeof renderFlightGantt === 'function') renderFlightGantt({ skipPathPrep: true });
    }, 70);
  }
  if (!document._allocGanttGlobalDragEndBound) {
    document._allocGanttGlobalDragEndBound = true;
    document.addEventListener('dragend', function() {
      if (_allocGanttPreviewTimer) {
        clearTimeout(_allocGanttPreviewTimer);
        _allocGanttPreviewTimer = null;
      }
      var st = state;
      var ctx = st._allocGanttDrag;
      if (!ctx || !ctx.flightId) return;
      if (st._allocGanttDropHandled) {
        st._allocGanttDrag = null;
        st._allocGanttDropHandled = false;
        _allocGanttPreviewLastKey = '';
        return;
      }
      var f = st.flights.find(function(x) { return x.id === ctx.flightId; });
      if (f) {
        f.standId = ctx.prevStandId || null;
        if (f.token) f.token.apronId = ctx.prevApron != null ? ctx.prevApron : (ctx.prevStandId || null);
      }
      var ctxFid = ctx.flightId;
      var prevSt = ctx.prevStandId;
      st._allocGanttDrag = null;
      st._allocGanttDropHandled = false;
      _allocGanttPreviewLastKey = '';
      if (f && typeof renderFlightList === 'function') {
        var touched = [];
        if (prevSt) touched.push(prevSt);
        if (f.standId) touched.push(f.standId);
        renderFlightList(false, false, { scheduleMode: 'incremental', dirtyFlightIds: [ctxFid], touchedStandIds: touched, skipGanttRefresh: true });
      }
      if (typeof renderFlightGantt === 'function') renderFlightGantt({ skipPathPrep: true });
    });
  }

  function _ganttWireInteractions(ganttEl, st) {
    const newScrollCol = ganttEl.querySelector('.alloc-gantt-scroll-col');
    if (newScrollCol && !newScrollCol._allocWheelBound) {
      newScrollCol._allocWheelBound = true;
      newScrollCol.addEventListener('wheel', function(ev) {
        if (!ev.ctrlKey) return;
        ev.preventDefault();
        newScrollCol.scrollLeft += (ev.deltaY || ev.deltaX || 0);
      }, { passive: false });
    }
    if (!ganttEl._allocDropBound) {
      ganttEl._allocDropBound = true;
      ganttEl.addEventListener('dragover', function(ev) {
        if (!ev.target || !ev.target.closest) return;
        if (!ev.target.closest('#allocationGantt')) return;
        const sc = ganttEl.querySelector('.alloc-gantt-scroll-col');
        if (!sc) return;
        const rect = sc.getBoundingClientRect();
        const x = Math.max(rect.left + 1, Math.min(rect.right - 1, ev.clientX));
        const el = document.elementFromPoint(ev.clientX, ev.clientY);
        let track = el && el.closest ? el.closest('.alloc-row-track') : null;
        if (!track && el && el.closest) {
          const row = el.closest('.alloc-row');
          if (row) track = row.querySelector ? row.querySelector('.alloc-row-track') : null;
        }
        if (!track) track = _ganttFindTrackAtPoint(sc, x, ev.clientY);
        ganttEl._lastDropTrack = track || null;
        if (track && track.getAttribute('data-apron-link-ok') === '0') {
          ev.preventDefault();
          ev.dataTransfer.dropEffect = 'none';
          return;
        }
        if (st._allocGanttDrag && st._allocGanttDrag.flightId) {
          var candPrev = null;
          if (track && track.getAttribute('data-runway-legend') !== '1')
            candPrev = track.getAttribute('data-stand-id') || null;
          _scheduleAllocGanttDragSchedulePreview(st, candPrev);
        }
        if (!ev.target.closest('.alloc-row-track')) {
          ev.preventDefault();
          ev.dataTransfer.dropEffect = 'move';
        }
      }, true);
      ganttEl.addEventListener('drop', function(ev) {
        if (!ev.target || !ev.target.closest) return;
        if (!ev.target.closest('#allocationGantt')) return;
        ev.preventDefault();
        ev.stopPropagation();
        const sc = ganttEl.querySelector('.alloc-gantt-scroll-col');
        if (!sc) return;
        let track = (ev.target && ev.target.closest('.alloc-row-track')) || null;
        if (!track) {
          const el = document.elementFromPoint(ev.clientX, ev.clientY);
          track = el && el.closest ? el.closest('.alloc-row-track') : null;
        }
        if (!track) track = ganttEl._lastDropTrack;
        if (!track) {
          const rect = sc.getBoundingClientRect();
          track = _ganttFindTrackAtPoint(sc, Math.max(rect.left + 1, Math.min(rect.right - 1, ev.clientX)), ev.clientY);
        }
        if (!track) return;
        if (track.getAttribute('data-runway-legend') === '1') return;
        if (track.getAttribute('data-apron-link-ok') === '0') return;
        const flightId = ev.dataTransfer.getData('text/plain');
        if (!flightId) return;
        const f = st.flights.find(function(x) { return x.id === flightId; });
        if (!f) return;
        if (!assignStandToFlight(f, track.getAttribute('data-stand-id') || null)) return;
        st._allocGanttDropHandled = true;
      }, true);
    }
    if (!ganttEl._allocZoomBound) {
      ganttEl._allocZoomBound = true;
      ganttEl.addEventListener('wheel', function(e) {
        if (!e.shiftKey) return;
        e.preventDefault();
        const factor = e.deltaY < 0 ? 1.15 : (1 / 1.15);
        let z = st.allocTimeZoom || 1;
        z = Math.max(1, Math.min(8, z * factor));
        st.allocTimeZoom = z;
        if (typeof renderFlightGantt === 'function') renderFlightGantt({ skipPathPrep: true });
      }, { passive: false });
    }
    ganttEl.querySelectorAll('.alloc-flight').forEach(function(el) {
      el.addEventListener('dragstart', function(ev) {
        var flightId = this.getAttribute('data-flight-id') || '';
        ev.dataTransfer.setData('text/plain', flightId);
        ev.dataTransfer.effectAllowed = 'move';
        var fDrag = st.flights.find(function(x) { return x.id === flightId; });
        if (fDrag) {
          st._allocGanttDragSeq = (st._allocGanttDragSeq || 0) + 1;
          st._allocGanttDrag = {
            flightId: flightId,
            prevStandId: fDrag.standId || null,
            prevApron: (fDrag.token && fDrag.token.apronId) ? fDrag.token.apronId : null,
            seq: st._allocGanttDragSeq
          };
          st._allocGanttDropHandled = false;
          _allocGanttPreviewLastKey = '';
        }
      });
      el.addEventListener('click', function(ev) {
        ev.stopPropagation();
        const flightId = this.getAttribute('data-flight-id');
        if (!flightId) return;
        const f = st.flights.find(function(x) { return x.id === flightId; });
        if (!f) return;
        state.flightPathRevealFlightId = null;
        st.selectedObject = { type: 'flight', id: flightId, obj: f };
        if (typeof updateObjectInfo === 'function') updateObjectInfo();
        if (typeof syncPanelFromState === 'function') syncPanelFromState();
        if (typeof draw === 'function') draw();
        const listEl = document.getElementById('flightList');
        if (listEl) {
          listEl.querySelectorAll('.obj-item').forEach(function(r) { r.classList.remove('selected', 'expanded'); });
          const row = listEl.querySelector('.obj-item[data-id="' + flightId + '"]');
          if (row) row.classList.add('selected', 'expanded');
        }
        if (typeof syncAllocGanttSelectionHighlight === 'function') syncAllocGanttSelectionHighlight();
      });
      el.addEventListener('dblclick', function(ev) {
        ev.stopPropagation();
        ev.preventDefault();
        const flightId = this.getAttribute('data-flight-id');
        if (!flightId) return;
        const f = st.flights.find(function(x) { return x.id === flightId; });
        if (!f) return;
        st.selectedObject = { type: 'flight', id: flightId, obj: f };
        state.flightPathRevealFlightId = flightId;
        if (typeof updateObjectInfo === 'function') updateObjectInfo();
        if (typeof syncPanelFromState === 'function') syncPanelFromState();
        if (typeof draw === 'function') draw();
        const listEl2 = document.getElementById('flightList');
        if (listEl2) {
          listEl2.querySelectorAll('.obj-item').forEach(function(r) { r.classList.remove('selected', 'expanded'); });
          const row2 = listEl2.querySelector('.obj-item[data-id="' + flightId + '"]');
          if (row2) row2.classList.add('selected', 'expanded');
        }
        if (typeof syncAllocGanttSelectionHighlight === 'function') syncAllocGanttSelectionHighlight();
      });
    });
    ganttEl.querySelectorAll('.alloc-row-track').forEach(function(track) {
      track.addEventListener('dragover', function(ev) {
        if (this.getAttribute('data-runway-legend') === '1') return;
        if (this.getAttribute('data-apron-link-ok') === '0') {
          ev.preventDefault();
          ev.dataTransfer.dropEffect = 'none';
          return;
        }
        ev.preventDefault();
        ev.dataTransfer.dropEffect = 'move';
      });
      track.addEventListener('drop', function(ev) {
        ev.preventDefault();
        if (this.getAttribute('data-runway-legend') === '1') return;
        if (this.getAttribute('data-apron-link-ok') === '0') return;
        const flightId = ev.dataTransfer.getData('text/plain');
        if (!flightId) return;
        const f = st.flights.find(function(x) { return x.id === flightId; });
        if (!f) return;
        if (!assignStandToFlight(f, this.getAttribute('data-stand-id') || null)) return;
        st._allocGanttDropHandled = true;
      });
    });
  }

  function validateNetworkInfrastructureOnly() {
    const msgs = [];
    const hasRunwayPath = state.taxiways && state.taxiways.some(tw => tw.pathType === 'runway');
    if (!hasRunwayPath) msgs.push('RunwayThere is no.');
    if (!state.taxiways || !state.taxiways.length) msgs.push('TaxiwayThere is no.');
    const stands = (state.pbbStands || []).concat(state.remoteStands || []);
    const linked = state.apronLinks || [];
    const hasApronLink = stands.some(pbb =>
      linked.some(lk =>
        lk.pbbId === pbb.id &&
        state.taxiways &&
        state.taxiways.some(tw => tw.id === lk.taxiwayId)
      )
    );
    if (!stands.length || !hasApronLink) msgs.push('Apron(PBB)class TaxiwayAt least one link is required to connect.');
    return msgs;
  }
  function validateNetworkForFlights() {
    const msgs = validateNetworkInfrastructureOnly();
    const termsForLabel = makeUniqueNamedCopy(state.terminals || [], 'name').map(function(t) { return {
      id: t.id,
      name: (t.name || '').trim() || 'Building'
    }; });
    function termNameById(id) {
      const tt = termsForLabel.find(function(t) { return t.id === id; });
      return tt ? tt.name : (id || 'Building');
    }
    const allStands = allStandsForFlightAssignment();
    (state.flights || []).forEach(function(f) {
      if (!f || !f.standId) return;
      const stand = allStands.find(function(s) { return s.id === f.standId; });
      if (!stand) return;
      const isRemoteOrTemp = (state.remoteStands || []).some(function(r) { return r.id === stand.id; })
        || (state.tempStands || []).some(function(r) { return r.id === stand.id; });
      if (!isRemoteOrTemp) return;
      const termId = (f.token && f.token.terminalId) || null;
      if (!termId) return;
      const allowed = Array.isArray(stand.allowedTerminals) ? stand.allowedTerminals : [];
      if (allowed.length && !allowed.includes(termId)) {
        const flightLabel = f.id || f.flightNo || f.reg || '';
        const standLabel = stand.name || 'Stand';
        const termLabel = termNameById(termId);
        const allowedLabel = allowed.map(termNameById).join(', ');
        msgs.push('Flight ' + (flightLabel || '') + ' building setting(' + termLabel + ') does not match stand ' + standLabel + ' available building settings (' + allowedLabel + ').');
      }
    });
    return msgs;
  }

  function updateFlightError(msgs) {
    const el = document.getElementById('flightError');
    if (!el) return;
    el.textContent = Array.isArray(msgs) ? msgs.join(' / ') : (msgs || '');
  }

  const REVERSE_COST = (function() {
    const v = Number((PATH_SEARCH_CFG || {}).reverseCost);
    return (isFinite(v) && v > 0) ? v : 1000000;
  })();
  function pathDist(a, b) { return Math.hypot(a[0]-b[0], a[1]-b[1]); }

  function clamp(v, min, max) {
    return Math.max(min, Math.min(max, v));
  }
  function sampleNormal(mu, sigma) {
    const u1 = Math.random() || 1e-9;
    const u2 = Math.random() || 1e-9;
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    return mu + sigma * z;
  }

  function pathPointKey(p) {
    const cs = (typeof CELL_SIZE === 'number' && CELL_SIZE > 0) ? CELL_SIZE : 20;
    const cellCol = Math.round(p[0] / cs * 2) / 2;
    const cellRow = Math.round(p[1] / cs * 2) / 2;
    return cellCol + ',' + cellRow;
  }

  function kpiToNumber(value) {
    const n = Number(value);
    return isFinite(n) ? n : null;
  }

  function kpiRound(value, digits) {
    const n = kpiToNumber(value);
    if (n == null) return null;
    const pow = Math.pow(10, digits || 0);
    return Math.round(n * pow) / pow;
  }

  function kpiFormatCount(value) {
    const n = kpiToNumber(value);
    return n == null ? '—' : String(Math.round(n));
  }

  function _kpiDurationSeconds(value, unit) {
    const n = kpiToNumber(value);
    if (n == null) return null;
    return unit === 'minutes' ? Math.max(0, Math.round(n * 60)) : Math.max(0, Math.round(n));
  }

  function _kpiFormatCompactDuration(totalSec, allowHours) {
    if (totalSec == null) return '—';
    const hours = Math.floor(totalSec / 3600);
    const mins = Math.floor((totalSec % 3600) / 60);
    const secs = totalSec % 60;
    if (allowHours && hours > 0) return hours + 'h ' + mins + 'm';
    if (mins > 0) return mins + 'm' + (secs > 0 ? ' ' + secs + 's' : (allowHours ? '' : ' 0s'));
    return secs + 's';
  }

  function _kpiFormatValueWithUnit(value, digits, unitLabel) {
    const n = kpiToNumber(value);
    if (n == null) return '—';
    return (digits > 0 ? n.toFixed(digits) : kpiRound(n, digits)) + ' ' + unitLabel;
  }

  function kpiFormatMinutesCompact(value) {
    return _kpiFormatCompactDuration(_kpiDurationSeconds(value, 'minutes'), true);
  }

  function kpiFormatSecondsCompact(value) {
    return _kpiFormatCompactDuration(_kpiDurationSeconds(value, 'seconds'), false);
  }

  function kpiFormatMinutesValue(value) {
    return _kpiFormatValueWithUnit(value, 1, 'min');
  }

  function kpiFormatSecondsValue(value) {
    return _kpiFormatValueWithUnit(value, 0, 'sec');
  }

  function kpiFormatClockBucket(minute) {
    const n = kpiToNumber(minute);
    if (n == null) return '—';
    const total = Math.floor(n);
    const hh = ((Math.floor(total / 60) % 24) + 24) % 24;
    return String(hh).padStart(2, '0') + ':00';
  }
  
  function kpiFormatClockBucket15(minute) {
    const n = kpiToNumber(minute);
    if (n == null) return '—';
    const total = Math.floor(n);
    const hh = ((Math.floor(total / 60) % 24) + 24) % 24;
    const mm = ((total % 60) + 60) % 60;
    return String(hh).padStart(2, '0') + ':' + String(mm).padStart(2, '0');
  }
  function kpiMinuteOfDay(t) {
    const n = kpiToNumber(t);
    if (n == null || !isFinite(n)) return null;
    const m = Math.floor(n);
    return ((m % 1440) + 1440) % 1440;
  }
  function kpiRollWindowOverlapsInterval(w, winMin, startMod, endMod) {
    if (startMod == null || endMod == null) return false;
    const winEnd = w + winMin;
    function segOverlap(a0, a1, b0, b1) {
      return a1 > b0 && a0 < b1;
    }
    if (endMod > startMod) return segOverlap(startMod, endMod, w, winEnd);
    if (endMod === startMod) return false;
    return segOverlap(startMod, 1440, w, winEnd) || segOverlap(0, endMod, w, winEnd);
  }

  function kpiFormatClock(minute) {
    const n = kpiToNumber(minute);
    if (n == null) return '—';
    return formatMinutesToHHMMSS(n);
  }

  function kpiFormatSnapshotTime() {
    const now = new Date();
    const hh = String(now.getHours()).padStart(2, '0');
    const mm = String(now.getMinutes()).padStart(2, '0');
    const ss = String(now.getSeconds()).padStart(2, '0');
    return hh + ':' + mm + ':' + ss;
  }

  function kpiSum(items, selector) {
    return (items || []).reduce(function(acc, item) {
      const value = selector(item);
      return acc + (kpiToNumber(value) || 0);
    }, 0);
  }

  function kpiAverage(items, selector) {
    const vals = (items || []).map(selector).map(kpiToNumber).filter(v => v != null);
    if (!vals.length) return null;
    return kpiSum(vals, function(v) { return v; }) / vals.length;
  }

  function kpiStandLabelById(standId) {
    const stands = allStandsForFlightAssignment();
    const stand = stands.find(function(s) { return s && s.id === standId; });
    return stand ? ((stand.name && stand.name.trim()) || stand.id || 'Stand') : 'Unassigned';
  }

  function kpiBuildMetricRow(label, primary, secondary) {
    return '' +
      '<div class="kpi-metric-row">' +
        '<div class="kpi-metric-label">' + escapeHtml(label) + '</div>' +
        '<div class="kpi-metric-values">' +
          '<div class="kpi-metric-primary">' + escapeHtml(primary) + '</div>' +
          '<div class="kpi-metric-secondary">' + escapeHtml(secondary) + '</div>' +
        '</div>' +
      '</div>';
  }

  function kpiBuildSummaryCard(label, value, tone) {
    return '' +
      '<div class="kpi-card ' + escapeHtml(tone || '') + '">' +
        '<div class="kpi-card-label">' + escapeHtml(label) + '</div>' +
        '<div class="kpi-card-value">' + escapeHtml(value) + '</div>' +
      '</div>';
  }

  function kpiBuildPanel(title, badge, rows) {
    return '' +
      '<div class="kpi-panel">' +
        '<div class="kpi-panel-header">' +
          '<div class="kpi-panel-title">' + escapeHtml(title) + '</div>' +
          '<div class="kpi-panel-badge">' + escapeHtml(badge) + '</div>' +
        '</div>' +
        '<div class="kpi-metric-list">' + rows.join('') + '</div>' +
      '</div>';
  }

  function kpiBucketOnHour(bucket) {
    const bs = kpiToNumber(bucket && bucket.bucketStart);
    if (bs == null || !isFinite(bs)) return false;
    const im = Math.floor(bs);
    return (im % 60 + 60) % 60 === 0;
  }
  function kpiDisposeInteractiveCharts() {
    try {
      if (window.__kpiChartGate) { window.__kpiChartGate.destroy(); window.__kpiChartGate = null; }
      if (window.__kpiChartRunway) { window.__kpiChartRunway.destroy(); window.__kpiChartRunway = null; }
    } catch (e) { console.warn('kpiDisposeInteractiveCharts', e); }
  }
  function kpiChartCommonOptions(buckets) {
    return {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { labels: { color: '#94a3b8', font: { size: 12, family: 'var(--ui-font, system-ui, sans-serif)' } } },


        tooltip: {
          backgroundColor: 'rgba(15, 23, 42, 0.94)',
          titleColor: '#f1f5f9',
          bodyColor: '#e2e8f0',
          borderColor: 'rgba(148, 163, 184, 0.28)',
          borderWidth: 1,
          padding: 10,
          callbacks: {
            title: function(items) {
              const i = items && items[0] ? items[0].dataIndex : 0;
              const b = buckets[i];
              if (!b) return '';
              const w = b.bucketStart != null ? kpiFormatClockBucket15(b.bucketStart) : (b.label || '');
              return 'w = ' + w + ' (60m rolling from w)';
            }
          }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255,255,255,0.07)' },
          ticks: {
            color: '#94a3b8',
            maxRotation: buckets.length > 24 ? 40 : 0,
            autoSkip: buckets.length > 36,
            maxTicksLimit: buckets.length > 36 ? 20 : undefined,
            font: { size: 12 },
            callback: function(tickValue, idx) {
              let i = idx;
              if (typeof tickValue === 'number' && isFinite(tickValue) && tickValue >= 0 && tickValue < buckets.length) {
                i = Math.round(tickValue);
              }
              const b = buckets[i];
              if (!b || !kpiBucketOnHour(b)) return '';
              return kpiFormatClockBucket(b.bucketStart);
            }
          }
        },
        y: {
          beginAtZero: true,
          grid: { color: 'rgba(255,255,255,0.07)' },
          ticks: { color: '#94a3b8', precision: 0, font: { size: 12 } }
        }
      }
    };
  }
  function kpiMountInteractiveCharts(buckets) {
    if (typeof Chart === 'undefined') {
      console.warn('Chart.js failed to load; KPI charts are static until CDN is available.');
      return;
    }
    if (!buckets || !buckets.length) return;
    const labels = buckets.map(function(b) { return b.label || kpiFormatClockBucket15(b.bucketStart); });
    const occ = buckets.map(function(b) { return b.occupancy || 0; });
    const arr = buckets.map(function(b) { return b.arrivals || 0; });
    const dep = buckets.map(function(b) { return b.departures || 0; });
    const tot = buckets.map(function(b) { return b.total || 0; });
    const opt = kpiChartCommonOptions(buckets);
    const elG = document.getElementById('kpiChartGateOcc');
    if (elG) {
      window.__kpiChartGate = new Chart(elG, {
        type: 'line',
        data: {
          labels: labels,
          datasets: [{
            label: 'Gate occupancy',
            data: occ,
            borderColor: '#a78bfa',
            backgroundColor: 'rgba(167, 139, 250, 0.22)',
            fill: true,
            tension: 0.28,
            pointRadius: 3,
            pointHoverRadius: 7,
            pointBackgroundColor: '#ddd6fe'
          }]
        },
        options: opt
      });
    }
    const elR = document.getElementById('kpiChartRunway');
    if (elR) {
      window.__kpiChartRunway = new Chart(elR, {
        type: 'bar',
        data: {
          labels: labels,
          datasets: [
            {
              type: 'bar',
              label: 'Runway arr (ELDT)',
              data: arr,
              backgroundColor: 'rgba(56, 189, 248, 0.72)',
              order: 3
            },
            {
              type: 'bar',
              label: 'Runway dep (ETOT)',
              data: dep,
              backgroundColor: 'rgba(251, 146, 60, 0.72)',
              order: 3
            },
            {
              type: 'line',
              label: 'Total',
              data: tot,
              borderColor: '#c4b5fd',
              backgroundColor: 'transparent',
              borderWidth: 3,
              tension: 0.22,
              pointRadius: 3,
              pointHoverRadius: 6,
              order: 1
            }
          ]
        },
        options: opt
      });
    }
  }
