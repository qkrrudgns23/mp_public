      const stand = allStands.find(function(s) { return s.id === standId; });
      if (!flightCanUseStand(f, stand)) {
        alert("Stand constraints or selected building do not match this aircraft, so it cannot be assigned.");
        return false;
      }
    }
    const prevStandForSched = f.standId || null;
    f.standId = standId;
    if (f.token) f.token.apronId = standId;
    delete f.sobtMin_orig;
    delete f.sldtMin_orig;
    delete f.sibtMin_orig;
    delete f.stotMin_orig;
    delete f.eldtMin_orig;
    delete f.eibtMin_orig;
    delete f.eobtMin_orig;
    delete f.etotMin_orig;
    if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
    const touchedSt = [];
    if (prevStandForSched) touchedSt.push(prevStandForSched);
    if (standId) touchedSt.push(standId);
    if (typeof renderFlightList === 'function')
      renderFlightList(false, false, { scheduleMode: 'incremental', dirtyFlightIds: [f.id], touchedStandIds: touchedSt });
    if (typeof draw === 'function') draw();
    return true;
  }

  function getCandidatePbbStandsForCode(code, flight) {
    const list = [];
    const allStands = (state.pbbStands || []).concat(state.remoteStands || []);
    allStands.forEach(stand => {
      if (flight && !flightCanUseStand(flight, stand)) return;
      if (!flight && code && getStandCategoryMode(stand) === 'icao' && stand.category && stand.category !== code) return;
      const hasLink = state.apronLinks.some(lk => lk.pbbId === stand.id);
      if (!hasLink) return;
      list.push(stand);
    });
    return list;
  }

  function pickRandom(arr) {
    if (!arr.length) return null;
    const idx = Math.floor(Math.random() * arr.length);
    return arr[idx];
  }

  function resolveStand(flight) {
    const allStands = (state.pbbStands || []).concat(state.remoteStands || []);
    if (flight.standId) {
      return allStands.find(s => s.id === flight.standId) || null;
    }
    let candidates = getCandidatePbbStandsForCode(flight.code, flight);
    if (!candidates.length) return null;
    const termId = (flight.token && flight.token.terminalId) || null;
    if (termId) {
      const filtered = candidates.filter(st => {
        const allowed = Array.isArray(st.allowedTerminals) ? st.allowedTerminals : null;
        if (allowed && allowed.length) return allowed.includes(termId);
        const t = getTerminalForStand(st);
        return t && t.id === termId;
      });
      if (filtered.length) candidates = filtered;
    }
    const stand = pickRandom(candidates);
    if (stand) flight.standId = stand.id;
    return stand;
  }

  function buildArrivalTimelineFromPts(flight, pts) {
    if (!pts || pts.length < 2) return null;
    const sibtMin_d = flight.sibtMin_d != null ? flight.sibtMin_d : (flight.timeMin != null ? flight.timeMin : 0);
    const baseT = sibtMin_d * 60;
    const v = Math.max(1, typeof getTaxiwayAvgMoveVelocityForPath === 'function' ? getTaxiwayAvgMoveVelocityForPath(null) : 10);
    const timeline = [];
    let tAcc = baseT;
    timeline.push({ t: tAcc, x: pts[0][0], y: pts[0][1] });
    for (let i = 1; i < pts.length; i++) {
      const [x1,y1] = pts[i-1];
      const [x2,y2] = pts[i];
      const len = Math.hypot(x2-x1, y2-y1);
      const dt = len / v;
      tAcc += dt;
      timeline.push({ t: tAcc, x: x2, y: y2 });
    }
    const sobtMin_d = flight.sobtMin_d != null ? flight.sobtMin_d : (sibtMin_d + (flight.dwellMin != null ? flight.dwellMin : 0));
    const dwellSec = Math.max(0, (sobtMin_d - sibtMin_d) * 60);
    if (dwellSec > 0) {
      tAcc = sobtMin_d * 60;
      const last = timeline[timeline.length - 1];
      timeline.push({ t: tAcc, x: last.x, y: last.y });
    }
    return timeline;
  }

  function buildDepartureTimelineFromPts(flight, pts) {
    if (!pts || pts.length < 2) return null;
    const sobtMin_d = flight.sobtMin_d != null ? flight.sobtMin_d : (flight.timeMin != null ? flight.timeMin + (flight.dwellMin != null ? flight.dwellMin : 0) : 0);
    const baseT = sobtMin_d * 60;
    const v = Math.max(1, typeof getTaxiwayAvgMoveVelocityForPath === 'function' ? getTaxiwayAvgMoveVelocityForPath(null) : 10);
    const timeline = [];
    let tAcc = baseT;
    timeline.push({ t: tAcc, x: pts[0][0], y: pts[0][1] });
    for (let i = 1; i < pts.length; i++) {
      const [x1,y1] = pts[i-1];
      const [x2,y2] = pts[i];
      const len = Math.hypot(x2-x1, y2-y1);
      const dt = len / v;
      tAcc += dt;
      timeline.push({ t: tAcc, x: x2, y: y2 });
    }
    return timeline;
  }

  function getFlightPositionAtTime(flight, tSec) {
    const tl = flight.timeline;
    if (!tl || !tl.length) return null;
    if (tSec < tl[0].t || tSec > tl[tl.length - 1].t) return null;
    for (let i = 0; i < tl.length - 1; i++) {
      const a = tl[i], b = tl[i+1];
      if (tSec >= a.t && tSec <= b.t) {
        const span = b.t - a.t || 1;
        const u = (tSec - a.t) / span;
        return {
          x: a.x + (b.x - a.x) * u,
          y: a.y + (b.y - a.y) * u
        };
      }
    }
    return null;
  }

  function getFlightPoseAtTime(flight, tSec) {
    const tl = flight.timeline;
    if (!tl || !tl.length) return null;
    if (tl.length === 1) {
      const a = tl[0];
      if (tSec + 1e-6 < a.t || tSec - 1e-6 > a.t) return null;
      const dg = a.deadlockGhost === true;
      if (a.motionForward === false) return { x: a.x, y: a.y, dx: -1, dy: 0, deadlockGhost: dg };
      return { x: a.x, y: a.y, dx: 1, dy: 0, deadlockGhost: dg };
    }
    if (tSec < tl[0].t || tSec > tl[tl.length - 1].t) return null;
    const motionChordEps = 0.08;
    const motionChordEps2 = motionChordEps * motionChordEps;
    function segmentUnitDir(segIdx) {
      if (segIdx < 0 || segIdx > tl.length - 2) return null;
      const p = tl[segIdx], q = tl[segIdx + 1];
      const ddx = q.x - p.x, ddy = q.y - p.y;
      const l2 = ddx * ddx + ddy * ddy;
      if (l2 < motionChordEps2) return null;
      const inv = 1 / Math.sqrt(l2);
      return { dx: ddx * inv, dy: ddy * inv };
    }
    function lastMotionUnitDirBefore(i) {
      for (let j = i - 1; j >= 0; j--) {
        const u = segmentUnitDir(j);
        if (u) return u;
      }
      return null;
    }
    function firstMotionUnitDirFrom(startSeg) {
      for (let j = startSeg; j <= tl.length - 2; j++) {
        const u = segmentUnitDir(j);
        if (u) return u;
      }
      return null;
    }
    function headingForInterval(i) {
      const a = tl[i], b = tl[i + 1];
      const dx = b.x - a.x, dy = b.y - a.y;
      const l2 = dx * dx + dy * dy;
      if (l2 >= motionChordEps2) return { dx: dx, dy: dy };
      const prev = lastMotionUnitDirBefore(i);
      if (prev) return { dx: prev.dx, dy: prev.dy };
      const next = firstMotionUnitDirFrom(i + 1);
      if (next) return { dx: next.dx, dy: next.dy };
      return { dx: 1, dy: 0 };
    }
    for (let i = 0; i < tl.length - 1; i++) {
      const a = tl[i], b = tl[i+1];
      if (tSec >= a.t && tSec <= b.t) {
        const span = b.t - a.t || 1;
        const u = (tSec - a.t) / span;
        const x = a.x + (b.x - a.x) * u;
        const y = a.y + (b.y - a.y) * u;
        const h = headingForInterval(i);
        const mfB = b.motionForward !== false;
        let rdx = h.dx, rdy = h.dy;
        if (mfB === false) {
          const len = Math.hypot(rdx, rdy) || 1;
          const ang = Math.atan2(rdy, rdx) + Math.PI;
          rdx = Math.cos(ang) * len;
          rdy = Math.sin(ang) * len;
        }
        const dg = !!(a.deadlockGhost || b.deadlockGhost);
        return { x, y, dx: rdx, dy: rdy, deadlockGhost: dg };
      }
    }
    return null;
  }

  
  function getFlightPoseAtTimeForDraw(flight, tSec) {
    const tl = flight && flight.timeline;
    if (!tl || !tl.length) return null;
    let t = Number(tSec);
    if (!isFinite(t)) return null;
    const t0 = tl[0].t, t1 = tl[tl.length - 1].t;
    if (t + 1e-9 < t0) return null;
    if (t > t1) t = t1;
    return getFlightPoseAtTime(flight, t);
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
  function syncFlightAssignStripFromFlight(f) {
    const arrEl = document.getElementById('flightAssignStripArr');
    const termEl = document.getElementById('flightAssignStripTerm');
    const depEl = document.getElementById('flightAssignStripDep');
    if (arrEl) {
      const sid = f ? (resolveArrivalRunwayIdForFlight(f) || '') : '';
      arrEl.innerHTML = buildRunwayOptionsHtml(sid);
      arrEl.value = sid;
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
    const sel = state.selectedObject;
    const hasFlight = sel && sel.type === 'flight' && sel.id;
    const f = hasFlight ? state.flights.find(function(x) { return x.id === sel.id; }) : null;
    const dis = !f;
    [arrEl, termEl, depEl].forEach(function(el) {
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
        var allStands = (st.pbbStands || []).concat(st.remoteStands || []);
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

  const FLIGHT_SCHED_TABLE_COL_COUNT = 27;
  /** tbody td index alignment with `_buildFlightListRowHtml` (0-based). S-group SLDT/STOT columns removed. */
  const FLIGHT_SCHED_TD_SLD = 10;
  const FLIGHT_SCHED_TD_SIBTD = 11;
  const FLIGHT_SCHED_TD_SOBTD = 12;
  const FLIGHT_SCHED_TD_STOTD = 13;
  const FLIGHT_SCHED_TD_ELDT = 14;
  const FLIGHT_SCHED_TD_EIBT = 15;
  const FLIGHT_SCHED_TD_EOBT = 16;
  const FLIGHT_SCHED_TD_ETOT = 17;
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
  function beginScheduleRetStatsBatch() {
    __schedRetStatsBatchActive = true;
    __schedRetStatsCached = null;
  }
  function endScheduleRetStatsBatch() {
    __schedRetStatsBatchActive = false;
    __schedRetStatsCached = null;
  }
  function getScheduleRetStatsAll() {
    if (__schedRetStatsBatchActive) {
      if (__schedRetStatsCached === null) {
        __schedRetStatsCached = typeof computeRunwayExitDistances === 'function' ? computeRunwayExitDistances() : [];
      }
      return __schedRetStatsCached;
    }
    return typeof computeRunwayExitDistances === 'function' ? computeRunwayExitDistances() : [];
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
    const arrDir = resolveArrivalRunwayDirForRetGate(f);
    return retStatsAll.some(function(r) {
      if (!r || !r.exit || r.exit.id !== f.sampledArrRet) return false;
      if (arrRunwayId == null) return true;
      if (!(r.runway && r.runway.id === arrRunwayId)) return false;
      if (arrDir === 'clockwise' || arrDir === 'counter_clockwise') {
        if (!isRunwayExitDirectionAllowed(r.exit, arrDir)) return false;
      }
      return true;
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
    const arrDir = resolveArrivalRunwayDirForRetGate(f);
    const candidates = retStatsAll.filter(function(r) {
      if (!(r && r.runway && r.runway.id === arrRunwayId && r.exit)) return false;
      if (arrDir === 'clockwise' || arrDir === 'counter_clockwise') {
        return isRunwayExitDirectionAllowed(r.exit, arrDir);
      }
      return true;
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
    const schedDepRotMin = Math.max(0, Number(SCHED_DEP_ROT_MIN) || 2);
    const sldtCalc = (f.sldtMin_d != null ? f.sldtMin_d : Math.max(0, tArrMin));
    const sldtOrig = f.sldtMin_orig != null ? f.sldtMin_orig : sldtCalc;
    const sobtOrig = (f.sobtMin_orig != null) ? f.sobtMin_orig : tDepMin;
    const stotOrig = (f.stotMin_orig != null) ? f.stotMin_orig : (tDepMin + schedDepRotMin);
    return {
      sibt: tArrMin,
      sobt: tDepMin,
      sldt_d: f.sldtMin_d != null ? f.sldtMin_d : sldtOrig,
      sibt_d: f.sibtMin_d != null ? f.sibtMin_d : tArrMin,
      sobt_d: f.sobtMin_d != null ? f.sobtMin_d : tDepMin,
      stot_d: f.stotMin_d != null ? f.stotMin_d : stotOrig,
    };
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
        '<th class="flight-col-sd flight-col-sd-start">SLDT(d)</th>' +
        '<th class="flight-col-sd">SIBT(d)</th>' +
        '<th class="flight-col-sd">SOBT(d)</th>' +
        '<th class="flight-col-sd flight-col-sd-last">STOT(d)</th>' +
        '<th class="flight-col-e flight-col-e-start">ELDT</th>' +
        '<th class="flight-col-e">EIBT</th>' +
        '<th class="flight-col-e">EOBT</th>' +
        '<th class="flight-col-e">ETOT</th>' +
        '<th class="flight-col-e flight-col-rot flight-th-mixed">ROT(arr)</th>' +
        '<th class="flight-th-mixed">STT(arr)</th>' +
        '<th class="flight-th-mixed">ATT(arr)</th>' +
        '<th class="flight-col-e flight-col-rot flight-th-mixed">ROT(dep)</th>' +
        '<th class="flight-th-mixed">STT(dep)</th>' +
        '<th class="flight-th-mixed">ATT(dep)</th>' +
        '<th>Aircraft Type</th>' +
        '<th class="flight-th-mixed">Code(ICAO)</th>' +
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
    const sldtStr_d = formatFlightScheduleDateTime(f, schedM.sldt_d);
    const sibtStr_d = formatFlightScheduleDateTime(f, schedM.sibt_d);
    const sobtStr_d = formatFlightScheduleDateTime(f, schedM.sobt_d);
    const stotStr_d = formatFlightScheduleDateTime(f, schedM.stot_d);
    function fmtFlightESchedCell(minVal) {
      return (typeof minVal === 'number' && isFinite(minVal)) ? formatMinutesToHHMMSS(minVal) : '—';
    }
    const eldtStr = fmtFlightESchedCell(f.eldtMin);
    const eibtStr = fmtFlightESchedCell(f.eibtMin);
    const eobtStr = fmtFlightESchedCell(f.eobtMin);
    const etotStr = fmtFlightESchedCell(f.etotMin);
    const dash = '—';
    const rotArrCell = (f.arrRotSec != null && isFinite(f.arrRotSec)) ? (Math.round(f.arrRotSec) + ' s') : dash;
    const rotDepCell = (f.depRotSec != null && isFinite(f.depRotSec)) ? (Math.round(f.depRotSec) + ' s') : dash;
    const sttArrCell = (typeof f.sttArrMin === 'number' && isFinite(f.sttArrMin)) ? formatMinutesToHHMMSS(f.sttArrMin) : dash;
    const attArrCell = (typeof f.attArrMin === 'number' && isFinite(f.attArrMin)) ? formatMinutesToHHMMSS(f.attArrMin) : dash;
    const sttDepCell = (typeof f.sttDepMin === 'number' && isFinite(f.sttDepMin)) ? formatMinutesToHHMMSS(f.sttDepMin) : dash;
    const attDepCell = (typeof f.attDepMin === 'number' && isFinite(f.attDepMin)) ? formatMinutesToHHMMSS(f.attDepMin) : dash;
    const depRunwayId = f.depRunwayId || (f.token && f.token.depRunwayId);
    const termId = f.terminalId || (f.token && f.token.terminalId);
    const arrRwRead = escapeHtml(getRunwayDisplayLabelById(arrRunwayId));
    const buildingRead = escapeHtml(getTerminalDisplayLabelById(termId));
    const depRwRead = escapeHtml(getRunwayDisplayLabelById(depRunwayId));
    const aircraftTypeLabel = ac ? (ac.name || ac.id || '') : (f.aircraftType || '—');
    const codeIcao = (ac && ac.icao) ? ac.icao : (f.code || '—');
    return '' +
      '<tr class="flight-data-row obj-item" data-id="' + f.id + '">' +
        '<td class="flight-td-reg">' + escapeHtml(f.reg || '') + '</td>' +
        '<td class="flight-td-reg">' + escapeHtml(f.airlineCode || '') + '</td>' +
        '<td class="flight-td-reg">' + escapeHtml(f.flightNumber || '') + '</td>' +
        '<td class="flight-td-readonly">' + arrRwRead + '</td>' +
        '<td class="flight-td-arr-ret' + (arrRetFailed ? ' flight-td-arr-ret-failed' : '') + '">' + (arrRetFailed ? 'Failed' : escapeHtml(sampledRetName)) + '</td>' +
        '<td class="flight-td-readonly">' + buildingRead + '</td>' +
        '<td class="flight-td-reg">' + (function() { var st = findStandById(f.standId); return escapeHtml(st ? ((st.name && st.name.trim()) || st.id || '—') : '—'); })() + '</td>' +
        '<td class="flight-td-readonly">' + depRwRead + '</td>' +
        '<td class="flight-td-time flight-col-s flight-col-s-start flight-td-sibt" data-sched-min="' + schedM.sibt + '">' + escapeHtml(sibtDisp) + '</td>' +
        '<td class="flight-td-time flight-col-s flight-col-s-last" data-sched-min="' + schedM.sobt + '">' + escapeHtml(sobtDisp) + '</td>' +
        '<td class="flight-td-time flight-col-sd flight-col-sd-start" data-sched-min="' + schedM.sldt_d + '">' + escapeHtml(sldtStr_d) + '</td>' +
        '<td class="flight-td-time flight-col-sd" data-sched-min="' + schedM.sibt_d + '">' + escapeHtml(sibtStr_d) + '</td>' +
        '<td class="flight-td-time flight-col-sd" data-sched-min="' + schedM.sobt_d + '">' + escapeHtml(sobtStr_d) + '</td>' +
        '<td class="flight-td-time flight-col-sd flight-col-sd-last" data-sched-min="' + schedM.stot_d + '">' + escapeHtml(stotStr_d) + '</td>' +
        '<td class="flight-td-time flight-col-e flight-col-e-start">' + eldtStr + '</td>' +
        '<td class="flight-td-time flight-col-e">' + eibtStr + '</td>' +
        '<td class="flight-td-time flight-col-e">' + eobtStr + '</td>' +
        '<td class="flight-td-time flight-col-e">' + etotStr + '</td>' +
        '<td class="flight-td-time flight-col-e flight-col-rot">' + rotArrCell + '</td>' +
        '<td class="flight-td-time">' + sttArrCell + '</td>' +
        '<td class="flight-td-time">' + attArrCell + '</td>' +
        '<td class="flight-td-time">' + rotDepCell + '</td>' +
        '<td class="flight-td-time">' + sttDepCell + '</td>' +
        '<td class="flight-td-time">' + attDepCell + '</td>' +
        '<td>' + escapeHtml(aircraftTypeLabel) + '</td>' +
        '<td>' + escapeHtml(codeIcao) + '</td>' +
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
  function _renderFlightListAfterPathEnsure(flightsSorted, schedFull, forceResampleRet, dirtySet, standSet, listEl, cfgEl) {
    if (forceResampleRet && typeof bumpVttArrCacheRev === 'function') bumpVttArrCacheRev();
    let retStatsAll = [];
    if (schedFull) {
      retStatsAll = (typeof ensureArrRetRotSampled === 'function')
        ? ensureArrRetRotSampled(flightsSorted, !!forceResampleRet)
        : (typeof computeRunwayExitDistances === 'function' ? computeRunwayExitDistances() : []);
    } else {
      const dirtyFlights = flightsSorted.filter(function(f) { return dirtySet.has(f.id); });
      const dirtyForRet = dirtyFlights.filter(function(f) { return f; });
      if (dirtyForRet.length && typeof ensureArrRetRotSampled === 'function')
        retStatsAll = ensureArrRetRotSampled(dirtyForRet, false);
      else
        retStatsAll = (typeof computeRunwayExitDistances === 'function') ? computeRunwayExitDistances() : [];
    }
    _renderFlightListDomAndSchedule(flightsSorted, schedFull, dirtySet, standSet, listEl, cfgEl, retStatsAll, null);
  }

  function renderFlightList(skipAutoAllocate, forceResampleRet, scheduleOpts, onDone) {
    const listEl = document.getElementById('flightList');
    const cfgEl = document.getElementById('flightConfigList');
    const cb = typeof onDone === 'function' ? onDone : null;
    if (!listEl) return;
    if (!state.flights.length) {
      _renderEmptyFlightListState(listEl, cfgEl);
      if (cb) cb();
      return;
    }
    if (scheduleOpts && scheduleOpts.pageTurnOnly === true && FLIGHT_SCHED_PAGE_SIZE > 0) {
      const flightsSorted = state.flights.slice();
      flightsSorted.sort((a, b) => (a.sibtMin_d != null ? a.sibtMin_d : (a.timeMin != null ? a.timeMin : 0)) - (b.sibtMin_d != null ? b.sibtMin_d : (b.timeMin != null ? b.timeMin : 0)));
      const retStatsAll = (typeof getScheduleRetStatsAll === 'function')
        ? getScheduleRetStatsAll()
        : ((typeof computeRunwayExitDistances === 'function') ? computeRunwayExitDistances() : []);
      _renderFlightListDomAndSchedule(flightsSorted, false, new Set(), new Set(), listEl, cfgEl, retStatsAll, { skipGanttRefresh: true });
      if (typeof syncAllocGanttSelectionHighlight === 'function') syncAllocGanttSelectionHighlight();
      if (cb) cb();
      return;
