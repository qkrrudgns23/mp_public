        const trailPts = getFlightTrailPolylineBackward(f, tSecDraw, FLIGHT_TRAIL_LENGTH_M);
        if (trailPts.length >= 2) {
          ctx.save();
          const x0 = trailPts[0][0], y0 = trailPts[0][1];
          const x1 = trailPts[trailPts.length - 1][0], y1 = trailPts[trailPts.length - 1][1];
          const g = ctx.createLinearGradient(x0, y0, x1, y1);
          const cFar = c2dSimFlightTrailStrokeEnd();
          const cNearAc = c2dSimFlightTrailStroke();
          g.addColorStop(0, cFar);
          g.addColorStop(0.42, cNearAc);
          g.addColorStop(1, cNearAc);
          ctx.strokeStyle = g;
          ctx.lineWidth = c2dSimFlightTrailLineWidth();
          ctx.lineCap = 'round';
          ctx.lineJoin = 'round';
          ctx.setLineDash([]);
          ctx.beginPath();
          ctx.moveTo(trailPts[0][0], trailPts[0][1]);
          for (let ti = 1; ti < trailPts.length; ti++) ctx.lineTo(trailPts[ti][0], trailPts[ti][1]);
          ctx.stroke();
          ctx.restore();
        }
      }
      if (isFlightPreTouchdownForDraw(f, tSecDraw)) {
        const rH = Math.max(sizeRef * 0.58, 8);
        ctx.save();
        ctx.beginPath();
        ctx.arc(x, y, rH, 0, Math.PI * 2);
        ctx.fillStyle = c2dSimPreTouchdownHaloFill();
        ctx.fill();
        ctx.strokeStyle = c2dSimPreTouchdownHaloStroke();
        ctx.lineWidth = 2;
        ctx.shadowColor = c2dSimPreTouchdownHaloStroke();
        ctx.shadowBlur = c2dSimPreTouchdownHaloBlur();
        ctx.stroke();
        ctx.restore();
      }
      if (isFlightSel) {
        ctx.save();
        ctx.beginPath();
        ctx.arc(x, y, sizeRef * 0.62, 0, Math.PI * 2);
        ctx.strokeStyle = c2dFlightSelectedRingStroke();
        ctx.lineWidth = 2.5;
        ctx.shadowColor = c2dFlightSelectedRingGlow();
        ctx.shadowBlur = c2dFlightSelectedRingGlowBlur();
        ctx.stroke();
        ctx.restore();
      }
      ctx.save();
      ctx.translate(x, y);
      const ang = Math.atan2(ny, nx);
      ctx.rotate(ang);
      const isDeadlockGhost = pose.deadlockGhost === true;
      ctx.fillStyle = isDeadlockGhost ? 'rgba(148, 163, 184, 0.45)' : apron2DGlyphFill();
      ctx.beginPath();
      if (useDetailSil) {
        ctx.moveTo(silhouette2D[0][0] * scaleX, silhouette2D[0][1] * scaleY);
        for (let si = 1; si < silhouette2D.length; si++) ctx.lineTo(silhouette2D[si][0] * scaleX, silhouette2D[si][1] * scaleY);
        ctx.closePath();
      } else {
        ctx.moveTo(scaleX * nX, 0);
        ctx.lineTo(scaleX * wRx, scaleY * uY);
        ctx.lineTo(scaleX * tX, 0);
        ctx.lineTo(scaleX * wRx, scaleY * lY);
        ctx.closePath();
      }
      ctx.fill();
      if (outlineWidth > 0 && outlineColor) {
        ctx.strokeStyle = isDeadlockGhost ? 'rgba(100, 116, 139, 0.55)' : outlineColor;
        ctx.lineWidth = outlineWidth;
        ctx.stroke();
      } else if (useDetailSil) {
        ctx.strokeStyle = isDeadlockGhost ? 'rgba(100, 116, 139, 0.5)' : 'rgba(15,23,42,0.85)';
        ctx.lineWidth = 1.15;
        ctx.stroke();
      }
      ctx.restore();
    });
    ctx.restore();
  }

  function ensureSimLoop() {
    if (ensureSimLoop._running) return;
    ensureSimLoop._running = true;
    ensureSimLoop._lastTs = null;
    function tick(ts) {
      let dt = 0;
      if (ensureSimLoop._lastTs != null) {
        dt = (ts - ensureSimLoop._lastTs) / 1000;
        if (dt < 0) dt = 0;
        if (dt > 0.25) dt = 0.25;
      }
      if (state.simPlaying && ensureSimLoop._playKick) {
        ensureSimLoop._playKick = false;
        dt = Math.max(dt, 1 / 60);
      }
      ensureSimLoop._lastTs = ts;
      if (state.simPlaying) {
        const lo = state.simStartSec, hi = state.simDurationSec;
        const speedRaw = state.simSpeed;
        const speed = (typeof speedRaw === 'number' && isFinite(speedRaw) && speedRaw > 0) ? speedRaw : 1;
        if (hi > lo + 1e-9) {
          state.simTimeSec = Math.min(state.simTimeSec + dt * speed, hi);
        } else {
          state.simTimeSec = lo;
        }
        const slider = document.getElementById('flightSimSlider');
        if (slider) slider.value = String(state.simTimeSec);
        updateFlightSimPlaybackLabelsDom();
        try { draw(); } catch(e) {}
        if (typeof update3DScene === 'function') update3DScene();
      }
      window.requestAnimationFrame(tick);
    }
    window.requestAnimationFrame(tick);
  }

  const AIRCRAFT_TYPES = (typeof INFORMATION === 'object' && INFORMATION && INFORMATION.tiers && INFORMATION.tiers.aircraft && Array.isArray(INFORMATION.tiers.aircraft.types)) ? INFORMATION.tiers.aircraft.types : [];
  const AIRCRAFT_BY_ID = {};
  AIRCRAFT_TYPES.forEach(a => { AIRCRAFT_BY_ID[a.id || a.name] = a; });
  function getAircraftInfoByType(typeId) {
    return AIRCRAFT_BY_ID[typeId] || null;
  }
  function getCodeForAircraft(typeId) {
    const a = getAircraftInfoByType(typeId);
    return (a && a.icao) ? a.icao : 'C';
  }
  function populateAircraftSelect(sel) {
    if (!sel) return;
    const opts = AIRCRAFT_TYPES.map(a => '<option value="' + escapeHtml(String(a.id || a.name || '')) + '">' + escapeHtml(a.name || a.id || '') + '</option>').join('');
    sel.innerHTML = opts || '<option value="A320">Airbus A320</option>';
    if (!opts && sel.options.length) sel.value = 'A320';
    else if (sel.options.length) {
      let hasA320 = false;
      for (let i = 0; i < sel.options.length; i++) {
        if (sel.options[i].value === 'A320') { hasA320 = true; break; }
      }
      sel.value = hasA320 ? 'A320' : sel.options[0].value;
    }
  }
  function getAircraftConstraintOptions() {
    return AIRCRAFT_TYPES.map(function(a) {
      const id = String(a.id || a.name || '').trim();
      const label = String(a.name || a.id || id || '').trim();
      return { id: id, label: label || id };
    }).filter(function(item) { return !!item.id; });
  }
  function normalizeStandCategoryMode(rawMode, fallbackMode) {
    const mode = String(rawMode || fallbackMode || 'icao').trim().toLowerCase();
    return mode === 'aircraft' ? 'aircraft' : 'icao';
  }
  function normalizeAllowedAircraftTypes(rawList) {
    const valid = new Set(getAircraftConstraintOptions().map(function(item) { return item.id; }));
    const out = [];
    (Array.isArray(rawList) ? rawList : []).forEach(function(item) {
      const id = String(item || '').trim();
      if (!id || !valid.has(id) || out.indexOf(id) >= 0) return;
      out.push(id);
    });
    return out;
  }
  function getStandCategoryMode(stand) {
    const isRemote = !!(stand && stand.x != null && stand.y != null && stand.x1 == null && stand.y1 == null);
    const fallback = isRemote ? (_remoteTier.defaultCategoryMode || 'icao') : (_pbbTier.defaultCategoryMode || 'icao');
    return normalizeStandCategoryMode(stand && stand.categoryMode, fallback);
  }
  function getStandAllowedAircraftTypes(stand) {
    return normalizeAllowedAircraftTypes(stand && stand.allowedAircraftTypes);
  }
  function getPbbLengthMeters(pbb) {
    const x1 = Number(pbb && pbb.x1), y1 = Number(pbb && pbb.y1);
    const x2 = Number(pbb && pbb.x2), y2 = Number(pbb && pbb.y2);
    if (Number.isFinite(x1) && Number.isFinite(y1) && Number.isFinite(x2) && Number.isFinite(y2)) {
      return Math.max(1, Math.hypot(x2 - x1, y2 - y1));
    }
    const anchor = getPbbAnchorPx(pbb);
    const center = getStandConnectionPx(pbb);
    return Math.max(1, Math.hypot(center[0] - anchor[0], center[1] - anchor[1]));
  }
  function getPbbAngleDeg(pbb) {
    return normalizeAngleDeg(getPBBStandAngle(pbb) * 180 / Math.PI);
  }
  function getStandConnectionPx(stand) {
    if (!stand) return [0, 0];
    if (stand.apronSiteX != null && stand.apronSiteY != null) return [Number(stand.apronSiteX), Number(stand.apronSiteY)];
    if (stand.x2 != null && stand.y2 != null) return [Number(stand.x2), Number(stand.y2)];
    if (stand.x != null && stand.y != null) return [Number(stand.x), Number(stand.y)];
    return cellToPixel(stand.col || 0, stand.row || 0);
  }
  function getStandRotationHandleRadiusPx() {
    return Math.max(6, CELL_SIZE * 0.22) * LAYOUT_VERTEX_DOT_SCALE;
  }
  function getPbbRotationOriginPx(pbb) {
    return getStandConnectionPx(pbb);
  }
  function getPbbRotationHandlePx(pbb) {
    const origin = getPbbRotationOriginPx(pbb);
    const safeAngle = getPBBStandAngle(pbb);
    const standSize = getStandSizeMeters((pbb && pbb.category) || 'C');
    const dist = getPbbLengthMeters(pbb) + Math.max(standSize * 0.55, 10);
    return [origin[0] + Math.cos(safeAngle) * dist, origin[1] + Math.sin(safeAngle) * dist];
  }
  function getRemoteRotationHandlePx(st) {
    const center = getRemoteStandCenterPx(st);
    const angle = getRemoteStandAngleRad(st);
    const standSize = getStandSizeMeters((st && st.category) || 'C');
    const dist = (standSize * 0.5) + Math.max(standSize * 0.35, 10);
    return [center[0] + Math.cos(angle) * dist, center[1] + Math.sin(angle) * dist];
  }
  function hitTestStandRotationHandle(wx, wy) {
    const maxD2 = Math.pow(getStandRotationHandleRadiusPx() * 1.9, 2);
    if (state.selectedObject && state.selectedObject.type === 'pbb' && state.selectedObject.obj) {
      const pbb = state.selectedObject.obj;
      const handle = getPbbRotationHandlePx(pbb);
      if (dist2(handle, [wx, wy]) <= maxD2) {
        return { type: 'pbb', id: pbb.id };
      }
    }
    if (state.selectedObject && state.selectedObject.type === 'remote' && state.selectedObject.obj) {
      const st = state.selectedObject.obj;
      const handle = getRemoteRotationHandlePx(st);
      if (dist2(handle, [wx, wy]) <= maxD2) {
        return { type: 'remote', id: st.id };
      }
    }
    return null;
  }
  function drawStandRotationHandle(originPx, handlePx, active) {
    if (!originPx || !handlePx) return;
    const r = getStandRotationHandleRadiusPx();
    ctx.save();
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = active ? '#ffffff' : 'rgba(255,255,255,0.65)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(originPx[0], originPx[1]);
    ctx.lineTo(handlePx[0], handlePx[1]);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = active ? '#f43f5e' : '#a78bfa';
    ctx.beginPath();
    ctx.arc(handlePx[0], handlePx[1], r, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }
  function buildDefaultPbbBridgePoints(pbb, bridgeIndex, bridgeCount) {
    const count = Math.max(1, parseInt(bridgeCount, 10) || 1);
    const length = getPbbLengthMeters(pbb);
    const angle = getPBBStandAngle(pbb);
    const dirX = Math.cos(angle), dirY = Math.sin(angle);
    const tanX = -dirY, tanY = dirX;
    const standSize = getStandSizeMeters((pbb && pbb.category) || 'C');
    const spread = Math.min(Math.max(standSize * 0.18, 4), standSize * 0.4);
    const offsetIndex = bridgeIndex - (count - 1) / 2;
    const lateral = spread * offsetIndex;
    const startX = Number(pbb.x1 || 0) + tanX * lateral;
    const startY = Number(pbb.y1 || 0) + tanY * lateral;
    const endX = Number(pbb.x2 || 0) + tanX * (lateral * 0.55);
    const endY = Number(pbb.y2 || 0) + tanY * (lateral * 0.55);
    const midX = startX + dirX * (length * 0.45);
    const midY = startY + dirY * (length * 0.45);
    return [
      { x: startX, y: startY },
      { x: midX, y: midY },
      { x: endX, y: endY },
    ];
  }
  function rebuildPbbBridgeGeometry(pbb) {
    const count = Math.max(1, Math.min(8, parseInt(pbb.pbbCount, 10) || 1));
    pbb.pbbCount = count;
    const prev = Array.isArray(pbb.pbbBridges) ? pbb.pbbBridges : [];
    pbb.pbbBridges = Array.from({ length: count }, function(_, idx) {
      const current = prev[idx];
      const points = (current && Array.isArray(current.points) && current.points.length >= 3)
        ? current.points.map(function(pt) { return { x: Number(pt.x) || 0, y: Number(pt.y) || 0 }; })
        : buildDefaultPbbBridgePoints(pbb, idx, count);
      return { id: (current && current.id) || id(), points: points };
    });
    if (pbb.apronSiteX == null || pbb.apronSiteY == null) {
      pbb.apronSiteX = Number(pbb.x2 || 0);
      pbb.apronSiteY = Number(pbb.y2 || 0);
    }
  }
  function setPbbGeometryFromAngleLength(pbb, angleDeg, lengthMeters, resetBridgeGeometry) {
    const ang = normalizeAngleDeg(angleDeg);
    const len = Math.max(1, Number(lengthMeters) || 1);
    const rad = ang * Math.PI / 180;
    const anchor = getPbbAnchorPx(pbb);
    pbb.x1 = anchor[0];
    pbb.y1 = anchor[1];
    pbb.x2 = anchor[0] + Math.cos(rad) * len;
    pbb.y2 = anchor[1] + Math.sin(rad) * len;
    pbb.angleDeg = ang;
    if (resetBridgeGeometry !== false) {
      delete pbb.pbbBridges;
    }
    rebuildPbbBridgeGeometry(pbb);
  }
  function normalizeBuildingObject(termLike) {
    const term = Object.assign({}, termLike || {});
    term.buildingType = normalizeBuildingType(term.buildingType || term.terminalType);
    if (Array.isArray(term.vertices)) {
      const cs = _persistCellSizePx();
      term.vertices = term.vertices.map(function(v) {
        if (!v || typeof v !== 'object') return { col: 0, row: 0 };
        const x = Number(v.x), y = Number(v.y);
        if (isFinite(x) && isFinite(y)) return { col: x / cs, row: y / cs };
        return { col: Number(v.col) || 0, row: Number(v.row) || 0 };
      });
    }
    return term;
  }
  function normalizePbbStandObject(rawPbb) {
    const pbb = Object.assign({}, rawPbb || {});
    pbb.categoryMode = getStandCategoryMode(pbb);
    pbb.allowedAircraftTypes = getStandAllowedAircraftTypes(pbb);
    pbb.pbbCount = Math.max(1, Math.min(8, parseInt(pbb.pbbCount != null ? pbb.pbbCount : (_pbbTier.defaultBridgeCount || 1), 10) || 1));
    if (pbb.x1 != null && pbb.y1 != null && pbb.x2 != null && pbb.y2 != null) {
      pbb.angleDeg = pbb.angleDeg != null
        ? normalizeAngleDeg(pbb.angleDeg)
        : normalizeAngleDeg(Math.atan2((Number(pbb.y2) || 0) - (Number(pbb.y1) || 0), (Number(pbb.x2) || 0) - (Number(pbb.x1) || 0)) * 180 / Math.PI);
      rebuildPbbBridgeGeometry(pbb);
    }
    return pbb;
  }
  function normalizeRemoteStandObject(rawStand) {
    const stand = Object.assign({}, rawStand || {});
    stand.categoryMode = getStandCategoryMode(stand);
    stand.allowedAircraftTypes = getStandAllowedAircraftTypes(stand);
    stand.angleDeg = normalizeAngleDeg(stand.angleDeg != null ? stand.angleDeg : 0);
    return stand;
  }

  (function initFlightUI() {
    (function wireFlightSchedulePagerOnce() {
      if (wireFlightSchedulePagerOnce._done) return;
      wireFlightSchedulePagerOnce._done = true;
      const bPrev = document.getElementById('btnFlightSchedPrev');
      const bNext = document.getElementById('btnFlightSchedNext');
      if (!bPrev || !bNext) return;
      bPrev.addEventListener('click', function() {
        if (FLIGHT_SCHED_PAGE_SIZE <= 0 || !state.flights.length) return;
        if (state.flightSchedulePage > 0) {
          state.flightSchedulePage--;
          renderFlightList(false, false, { pageTurnOnly: true });
        }
      });
      bNext.addEventListener('click', function() {
        if (FLIGHT_SCHED_PAGE_SIZE <= 0 || !state.flights.length) return;
        const nFl = state.flights.length;
        const maxP = Math.max(0, Math.ceil(nFl / FLIGHT_SCHED_PAGE_SIZE) - 1);
        if (state.flightSchedulePage < maxP) {
          state.flightSchedulePage++;
          renderFlightList(false, false, { pageTurnOnly: true });
        }
      });
    })();
    const arrDepEl = document.getElementById('flightArrDep');
    const dwellEl = document.getElementById('flightDwell');
    const minDwellEl = document.getElementById('flightMinDwell');
    const addBtn = document.getElementById('btnAddFlight');
    const playBtn = document.getElementById('btnPlayFlights');
    const pauseBtn = document.getElementById('btnPauseFlights');
    const resetBtn = document.getElementById('btnResetFlights');
    const simSlider = document.getElementById('flightSimSlider');
    const speedSelect = document.getElementById('flightSpeed');
    const timeInputEl = document.getElementById('flightTime');
    const sibtDateInputEl = document.getElementById('flightSibtDate');
    const aircraftEl = document.getElementById('flightAircraftType');
    const regEl = document.getElementById('flightReg');
    const layoutNameInput = document.getElementById('layoutName');
    const saveLayoutBtn = document.getElementById('btnSaveLayout');
    const layoutMsgEl = document.getElementById('layoutMessage');
    const layoutLoadListEl = document.getElementById('layoutLoadList');
    const globalUpdateBtn = document.getElementById('btnGlobalUpdate');
    if (!arrDepEl) return;
    populateAircraftSelect(aircraftEl);

    function normalizeSibtDate(raw) {
      const s = (raw == null ? '' : String(raw)).trim();
      if (/^\d{4}-\d{2}-\d{2}$/.test(s)) return s;
      return DEFAULT_SIBT_DATE;
    }
    function formatFlightScheduleGateInputs(f) {
      if (!f) return;
      const tArrMin = f.timeMin != null ? f.timeMin : 0;
      if (timeInputEl) timeInputEl.value = formatMinutesToHHMMSS(tArrMin);
      if (sibtDateInputEl) {
        const dateRaw = f.sibtDate != null ? f.sibtDate : (f.serviceDate != null ? f.serviceDate : DEFAULT_SIBT_DATE);
        sibtDateInputEl.value = normalizeSibtDate(dateRaw);
      }
    }
    function bindFlightScheduleGateFieldChange(el, field) {
      if (!el) return;
      el.addEventListener('change', function() {
        if (!state.selectedObject || state.selectedObject.type !== 'flight') return;
        const f = state.selectedObject.obj;
        const m = typeof parseTimeToMinutes === 'function' ? parseTimeToMinutes(this.value || '0') : NaN;
        if (!isFinite(m)) {
          formatFlightScheduleGateInputs(f);
          return;
        }
        if (typeof applyScheduledGateTimingFromSField !== 'function') {
          formatFlightScheduleGateInputs(f);
          return;
        }
        const ok = applyScheduledGateTimingFromSField(f, field, m);
        if (!ok) {
          formatFlightScheduleGateInputs(f);
          return;
        }
        if (dwellEl) dwellEl.value = f.dwellMin != null ? f.dwellMin : 0;
        if (minDwellEl) minDwellEl.value = f.minDwellMin != null ? f.minDwellMin : 0;
        formatFlightScheduleGateInputs(f);
        if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
        const touched = f.standId ? [f.standId] : [];
        if (typeof renderFlightList === 'function')
          renderFlightList(false, false, { scheduleMode: 'incremental', dirtyFlightIds: [f.id], touchedStandIds: touched });
        if (typeof draw === 'function') draw();
      });
    }
    bindFlightScheduleGateFieldChange(timeInputEl, 'sibt');
    if (sibtDateInputEl) {
      sibtDateInputEl.addEventListener('change', function() {
        if (!state.selectedObject || state.selectedObject.type !== 'flight') return;
        const f = state.selectedObject.obj;
        f.sibtDate = normalizeSibtDate(this.value);
        f.serviceDate = f.sibtDate;
        this.value = f.sibtDate;
        if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
        const touched = f.standId ? [f.standId] : [];
        if (typeof renderFlightList === 'function')
          renderFlightList(false, false, { scheduleMode: 'incremental', dirtyFlightIds: [f.id], touchedStandIds: touched });
        if (typeof draw === 'function') draw();
      });
    }

    function randomAirlineCode() { return DEFAULT_AIRLINE_CODES[Math.floor(Math.random() * DEFAULT_AIRLINE_CODES.length)]; }
    function randomFlightNumber(airlineCode) { return (airlineCode || randomAirlineCode()) + String(Math.floor(1000 + Math.random() * 9000)); }
    function getDefaultSibtMinutes() {
      let maxT = 0;


      (state.flights || []).forEach(f => {
        if (!f) return;
        const sibt = f.sibtMin_d != null ? f.sibtMin_d : (typeof f.timeMin === 'number' ? f.timeMin : 0);
        if (isFinite(sibt) && sibt > maxT) maxT = sibt;
      });
      return maxT + 10;
    }
    if (dwellEl) {
      const syncDwell = () => {
        const isArr = arrDepEl.value === 'Arr';
        dwellEl.disabled = !isArr;
        if (!isArr) dwellEl.value = dwellEl.value || 0;
      };
      arrDepEl.addEventListener('change', syncDwell);
      syncDwell();
    }
    if (minDwellEl) {
      const syncMinDwell = () => {
        const isArr = arrDepEl.value === 'Arr';
        minDwellEl.disabled = !isArr;
        if (!isArr) minDwellEl.value = minDwellEl.value || 0;
      };
      arrDepEl.addEventListener('change', syncMinDwell);
      syncMinDwell();
    }
    const TOKEN_NODE_ORDER = ['runway','taxiway','apron','terminal'];
    function fillTokenSelects(flightCode) {
      const runwaySel = document.getElementById('tokenRunwaySelect');
      const termSel = document.getElementById('tokenTerminalSelect');
      if (runwaySel) {
        const opts = getRunwayOptions();
        runwaySel.innerHTML = '<option value="">Random</option>' + opts.map(o => '<option value="' + (o.id || '').replace(/"/g, '&quot;') + '">' + (o.name || o.id || '').replace(/</g, '&lt;') + '</option>').join('');
      }
      if (termSel) {
        const terms = (state.terminals || []).map(t => ({ id: t.id, name: (t.name || '').trim() || 'Building' }));
        termSel.innerHTML = '<option value="">Random</option>' + terms.map(o => '<option value="' + (o.id || '').replace(/"/g, '&quot;') + '">' + (o.name || o.id || '').replace(/</g, '&lt;') + '</option>').join('');
      }
    }
    function updateTokenPanesVisibility(nodes) {
      const arr = Array.isArray(nodes) ? nodes : TOKEN_NODE_ORDER;
      ['runway','taxiway','apron','terminal'].forEach((node, i) => {
        const el = document.getElementById('tokenObject' + node.charAt(0).toUpperCase() + node.slice(1));
        if (el) el.style.display = arr.indexOf(node) >= 0 ? 'block' : 'none';
      });
    }
    function proSimApiBase() {
      if (LAYOUT_API_URL && LAYOUT_API_URL !== 'null') return LAYOUT_API_URL;
      try {
        if (window.location && window.location.origin && window.location.origin !== 'null') return window.location.origin;
      } catch (e) { /* ignore */ }
      return '';
    }
    if (globalUpdateBtn) {
      globalUpdateBtn.addEventListener('click', function() {
        function failProSim(msg) {
          const m = (msg && String(msg)) || 'Pro Sim failed';
          console.error('Pro Sim:', m);
          if (typeof setGlobalUpdateProgressUi === 'function') setGlobalUpdateProgressUi(false);
          if (typeof alert === 'function') alert(m);
        }
        const base = proSimApiBase();
        if (!base) {
          failProSim('Layout API가 설정되지 않았습니다. run_app.py로 서버를 띄운 뒤 다시 시도하세요.');
          return;
        }
        if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
        if (typeof clearAllFlightTimelines === 'function') clearAllFlightTimelines();
        const playDockBtnEl = document.getElementById('btnShowPlayDock');
        if (playDockBtnEl) playDockBtnEl.disabled = true;
        try {
          if (typeof syncStateFromPanel === 'function') syncStateFromPanel();
          if (typeof syncTableToFlightState === 'function') syncTableToFlightState();
        } catch (e0) {
          failProSim(e0 && e0.message);
          return;
        }
        const layoutName = (state.currentLayoutName && String(state.currentLayoutName).trim()) || INITIAL_LAYOUT_DISPLAY_NAME || 'default_layout';
        let layoutPayload;
        try {
          layoutPayload = serializeCurrentLayout();
        } catch (e1) {
          failProSim(e1 && e1.message);
          return;
        }
        if (typeof setGlobalUpdateProgressUi === 'function') {
          setGlobalUpdateProgressUi(true, 'airside_sim 시작…', 3);
        }
        fetch(base + '/api/run-simulation', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ layout: layoutPayload, layoutName: layoutName, name: layoutName }),
        }).then(function(r) {
          if (r.status === 409) {
            return r.json().then(function(d) {
              throw new Error((d && d.error) || '시뮬레이션이 이미 실행 중입니다.');
            });
          }
          if (!r.ok) {
            return r.text().then(function(t) {
              throw new Error(t || ('HTTP ' + r.status));
            });
          }
          return r.json();
        }).then(function() {
          function pollProgress() {
            fetch(base + '/api/sim-progress')
              .then(function(pr) { return pr.json(); })
              .then(function(p) {
                if (p && p.running) {
                  const pct = (p.percent != null && isFinite(Number(p.percent))) ? Number(p.percent) : 0;
                  if (typeof setGlobalUpdateProgressUi === 'function') {
                    setGlobalUpdateProgressUi(true, 'Airside DES (utils/airside_sim) 실행 중…', pct);
                  }
                  setTimeout(pollProgress, 350);
                  return;
                }
                if (p && p.error) {
                  failProSim(String(p.error));
                  return;
                }
                if (typeof setGlobalUpdateProgressUi === 'function') setGlobalUpdateProgressUi(false);
                const layoutNameDone = (state.currentLayoutName && String(state.currentLayoutName).trim()) || INITIAL_LAYOUT_DISPLAY_NAME || 'default_layout';
                fetch(base + '/api/load-sim-result?name=' + encodeURIComponent(layoutNameDone))
                  .then(function(r) {
                    if (!r.ok) throw new Error('시뮬 결과를 불러오지 못했습니다.');
                    return r.json();
                  })
                  .then(function(data) {
                    if (typeof applyAirsideSimulationResultPayload === 'function') applyAirsideSimulationResultPayload(data);
                  })
                  .catch(function(e) {
                    console.warn('Pro Sim result fetch', e && e.message ? e.message : e);
                  });
              })
              .catch(function(e) {
                failProSim(e && e.message ? e.message : 'sim-progress 요청 실패');
              });
          }
          pollProgress();
        }).catch(function(e) {
          failProSim(e && e.message ? e.message : String(e));
        });
      });
    }
    const btnShowPlayDock = document.getElementById('btnShowPlayDock');
    if (btnShowPlayDock) {
      btnShowPlayDock.addEventListener('click', function() {
        state.simPlaybackDockVisible = true;
        if (typeof applySimPlaybackBarDomVisibility === 'function') applySimPlaybackBarDomVisibility();
      });
    }
    function applyTokenNodesFromCheckboxes() {
      const nodes = [];
      TOKEN_NODE_ORDER.forEach((node, i) => {
        const cb = document.getElementById('token' + node.charAt(0).toUpperCase() + node.slice(1));
        if (cb && cb.checked) nodes.push(node);
        else return;
      });
      return nodes;
    }
    function setTokenCheckboxesFromNodes(nodes) {
      const arr = Array.isArray(nodes) ? nodes : [];
      TOKEN_NODE_ORDER.forEach((node, i) => {
        const cb = document.getElementById('token' + node.charAt(0).toUpperCase() + node.slice(1));
        if (cb) cb.checked = arr.indexOf(node) >= 0;
      });
      updateTokenPanesVisibility(arr.length ? arr : TOKEN_NODE_ORDER);
    }
    ['Runway','Taxiway','Apron','Building'].forEach((name, i) => {
      const cb = document.getElementById('token' + name);
      if (!cb) return;
      cb.addEventListener('change', function() {
        if (!state.selectedObject || state.selectedObject.type !== 'flight') return;
        const f = state.selectedObject.obj;
        if (!f.token) f.token = { nodes: TOKEN_NODE_ORDER.slice(), runwayId: null, apronId: null, terminalId: null };
        if (this.checked) {
          f.token.nodes = TOKEN_NODE_ORDER.slice(0, i + 1);
          setTokenCheckboxesFromNodes(f.token.nodes);
        } else {
          f.token.nodes = TOKEN_NODE_ORDER.slice(0, i);
          setTokenCheckboxesFromNodes(f.token.nodes);
        }
        updateTokenPanesVisibility(f.token.nodes);
        rebuildSelectedFlightTimeline();
      });
    });
    const tokenRunwaySel = document.getElementById('tokenRunwaySelect');
    const tokenTerminalSel = document.getElementById('tokenTerminalSelect');
    if (tokenRunwaySel) tokenRunwaySel.addEventListener('change', function() {
      if (!state.selectedObject || state.selectedObject.type !== 'flight') return;
      const f = state.selectedObject.obj;
      if (!f.token) f.token = { nodes: TOKEN_NODE_ORDER.slice(), runwayId: null, apronId: null, terminalId: null };
      f.token.runwayId = this.value || null;
      rebuildSelectedFlightTimeline();
    });
    if (tokenTerminalSel) tokenTerminalSel.addEventListener('change', function() {
      if (!state.selectedObject || state.selectedObject.type !== 'flight') return;
      const f = state.selectedObject.obj;
      if (!f.token) f.token = { nodes: TOKEN_NODE_ORDER.slice(), runwayId: null, apronId: null, terminalId: null };
      f.token.terminalId = this.value || null;
      rebuildSelectedFlightTimeline();
    });
    const flightSubtabButtons = document.querySelectorAll('.flight-subtab');
    const flightPaneSchedule = document.getElementById('flightPaneSchedule');
    const flightPaneConfig = document.getElementById('flightPaneConfig');
    if (flightSubtabButtons && flightPaneSchedule && flightPaneConfig) {
      flightSubtabButtons.forEach(btn => {
        btn.addEventListener('click', function() {
          const target = this.getAttribute('data-flight-subtab') || 'schedule';
          flightSubtabButtons.forEach(b => b.classList.remove('active'));
          this.classList.add('active');
          if (target === 'config') {
            flightPaneSchedule.style.display = 'none';
            flightPaneConfig.style.display = 'block';
          } else {
            flightPaneSchedule.style.display = 'block';
            flightPaneConfig.style.display = 'none';
          }
        });
      });
    }
    if (addBtn) {
      addBtn.addEventListener('click', function() {
        const networkErrors = validateNetworkForFlights();
        if (networkErrors.length) {
          updateFlightError(networkErrors);
          alert('Flightcannot be created:\\n' + networkErrors.join('\\n'));
          return;
        }
        let timeStr = (document.getElementById('flightTime').value || '').trim();
        if (!timeStr) {
          timeStr = formatMinutesToHHMMSS(DEFAULT_SIBT_TIME_MIN);
          if (timeInputEl) timeInputEl.value = timeStr;
        }
        const timeMin = parseTimeToMinutes(timeStr);
        const sibtDateForFlight = sibtDateInputEl ? normalizeSibtDate(sibtDateInputEl.value || DEFAULT_SIBT_DATE) : DEFAULT_SIBT_DATE;
        if (sibtDateInputEl) sibtDateInputEl.value = sibtDateForFlight;
        const aircraftType = (document.getElementById('flightAircraftType').value || 'A320').trim();
        const code = getCodeForAircraft(aircraftType);
        const reg = (document.getElementById('flightReg').value || '').trim();
        let airlineCode = (document.getElementById('flightAirlineCode') && document.getElementById('flightAirlineCode').value || '').trim();
        let flightNumber = (document.getElementById('flightFlightNumber') && document.getElementById('flightFlightNumber').value || '').trim();
        if (!airlineCode) airlineCode = randomAirlineCode();
        if (!flightNumber) flightNumber = randomFlightNumber(airlineCode);
        let dwellMin = parseFloat(document.getElementById('flightDwell').value);
        let minDwellMin = parseFloat(document.getElementById('flightMinDwell').value);
        dwellMin = (typeof dwellMin === 'number' && !isNaN(dwellMin) && dwellMin >= 0) ? dwellMin : 0;
        minDwellMin = (typeof minDwellMin === 'number' && !isNaN(minDwellMin) && minDwellMin >= 0) ? minDwellMin : 0;
        dwellMin = Math.max(SCHED_DWELL_FLOOR_MIN, dwellMin);
        minDwellMin = Math.max(SCHED_DWELL_FLOOR_MIN, minDwellMin);
        if (minDwellMin > dwellMin) minDwellMin = dwellMin;
        const arrDep = 'Arr';
        const runwayOptions = getRunwayOptions();
        const defaultRunwayId = runwayOptions.length ? (runwayOptions[0].id || null) : null;
        const f = {
          id: id(),
          arrDep,
          timeMin,
          sibtDate: sibtDateForFlight,
          serviceDate: sibtDateForFlight,
          aircraftType,
          code,
          reg,
          airlineCode,
          flightNumber,
          dwellMin,
          minDwellMin,
          arrRunwayId: defaultRunwayId,
          depRunwayId: defaultRunwayId,
          timeline: null,
          token: {
            nodes: ['runway','taxiway','apron','terminal'],
            runwayId: defaultRunwayId,
            arrRunwayId: defaultRunwayId,
            depRunwayId: defaultRunwayId,
            apronId: null,
            terminalId: null
          }
        };
        state.flights.push(f);
        if (typeof syncSimulationPlaybackAfterTimelines === 'function') syncSimulationPlaybackAfterTimelines();
        else if (typeof recomputeSimDuration === 'function') recomputeSimDuration();
        if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
        var addTouched = f.standId ? [f.standId] : [];
        renderFlightList(false, false, { scheduleMode: 'incremental', dirtyFlightIds: [f.id], touchedStandIds: addTouched });
        const nextDef = getDefaultSibtMinutes();
        if (timeInputEl) timeInputEl.value = formatMinutesToHHMMSS(nextDef);
        updateFlightError('');
      });
    }
    function syncFlightPanelFromSelection() {
      if (!state.selectedObject || state.selectedObject.type !== 'flight') return;
      const f = state.selectedObject.obj;
      if (arrDepEl) arrDepEl.value = 'Arr';
      if (dwellEl) {
        dwellEl.disabled = false;
        dwellEl.value = f.dwellMin || 0;
      }
      if (minDwellEl) {
        minDwellEl.disabled = false;
        minDwellEl.value = f.minDwellMin != null ? f.minDwellMin : 0;
      }
      formatFlightScheduleGateInputs(f);
      if (aircraftEl) {
        if (f.aircraftType && AIRCRAFT_BY_ID[f.aircraftType]) aircraftEl.value = f.aircraftType;
        else {
          const match = AIRCRAFT_TYPES.find(a => a.icao === (f.code || 'C'));
          aircraftEl.value = match ? match.id : (AIRCRAFT_TYPES[0] && AIRCRAFT_TYPES[0].id) || 'A320';
        }
      }
      if (regEl) regEl.value = f.reg || '';
      const airlineCodeEl = document.getElementById('flightAirlineCode');
      const flightNumberEl = document.getElementById('flightFlightNumber');
      if (airlineCodeEl) airlineCodeEl.value = f.airlineCode || '';
      if (flightNumberEl) flightNumberEl.value = f.flightNumber || '';
      if (!f.token) f.token = { nodes: TOKEN_NODE_ORDER.slice(), runwayId: null, apronId: null, terminalId: null };
      fillTokenSelects(f.code);
      setTokenCheckboxesFromNodes(f.token.nodes);
      if (tokenRunwaySel) tokenRunwaySel.value = f.token.runwayId || '';
      if (tokenTerminalSel) tokenTerminalSel.value = f.token.terminalId || '';
      if (typeof syncFlightAssignStrip === 'function') syncFlightAssignStrip();
    }
    hookSyncFlightPanelFromSelection = syncFlightPanelFromSelection;
    const origSyncPanel = syncPanelFromState;
    syncPanelFromState = function() {
      origSyncPanel();
      if (activeTab === 'flight') syncFlightPanelFromSelection();
    };
    function rebuildSelectedFlightTimeline() {
      if (!state.selectedObject || state.selectedObject.type !== 'flight') return;
      if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
      const f = state.selectedObject.obj;
      updateFlightError('');
      draw();
      var sidSched = f.standId || null;
      if (typeof renderFlightList === 'function')
        renderFlightList(false, false, { scheduleMode: 'incremental', dirtyFlightIds: [f.id], touchedStandIds: sidSched ? [sidSched] : [] });
    }
    if (arrDepEl) {
      arrDepEl.addEventListener('change', function() {
        if (!state.selectedObject || state.selectedObject.type !== 'flight') return;
        const f = state.selectedObject.obj;
        f.arrDep = this.value === 'Dep' ? 'Dep' : 'Arr';
        if (dwellEl) {
          dwellEl.disabled = f.arrDep !== 'Arr';
          if (f.arrDep !== 'Arr') {
            f.dwellMin = 0;
            dwellEl.value = 0;
          } else {
            f.dwellMin = parseFloat(dwellEl.value) || 0;
          }
        }
        if (minDwellEl) {
          minDwellEl.disabled = f.arrDep !== 'Arr';
          if (f.arrDep !== 'Arr') {
            f.minDwellMin = 0;
            minDwellEl.value = 0;
          } else {
            f.minDwellMin = Math.max(0, parseFloat(minDwellEl.value) || 0);
            minDwellEl.value = f.minDwellMin;
          }
        }
        rebuildSelectedFlightTimeline();
      });
    }
    if (aircraftEl) {
      aircraftEl.addEventListener('change', function() {
        if (!state.selectedObject || state.selectedObject.type !== 'flight') return;
        const f = state.selectedObject.obj;
        f.aircraftType = this.value || 'A320';
        f.code = getCodeForAircraft(f.aircraftType);
        rebuildSelectedFlightTimeline();
      });
    }
    if (regEl) {
      regEl.addEventListener('change', function() {
        if (!state.selectedObject || state.selectedObject.type !== 'flight') return;
        const f = state.selectedObject.obj;
        f.reg = this.value || '';
        if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
        var rs = f.standId || null;
        renderFlightList(false, false, { scheduleMode: 'incremental', dirtyFlightIds: [f.id], touchedStandIds: rs ? [rs] : [] });
        updateObjectInfo();
      });
    }
    const airlineCodeEl = document.getElementById('flightAirlineCode');
    const flightNumberEl = document.getElementById('flightFlightNumber');
    if (airlineCodeEl) {
      airlineCodeEl.addEventListener('change', function() {
        if (!state.selectedObject || state.selectedObject.type !== 'flight') return;
        const f = state.selectedObject.obj;
        f.airlineCode = this.value || '';
        if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
        var rs2 = f.standId || null;
        renderFlightList(false, false, { scheduleMode: 'incremental', dirtyFlightIds: [f.id], touchedStandIds: rs2 ? [rs2] : [] });
        updateObjectInfo();
      });
    }
    if (flightNumberEl) {
      flightNumberEl.addEventListener('change', function() {
        if (!state.selectedObject || state.selectedObject.type !== 'flight') return;
        const f = state.selectedObject.obj;
        f.flightNumber = this.value || '';
        if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
        var rs3 = f.standId || null;
        renderFlightList(false, false, { scheduleMode: 'incremental', dirtyFlightIds: [f.id], touchedStandIds: rs3 ? [rs3] : [] });
        updateObjectInfo();
      });
    }
    if (dwellEl) {
      dwellEl.addEventListener('change', function() {
        if (!state.selectedObject || state.selectedObject.type !== 'flight') return;
        const f = state.selectedObject.obj;
        let v = parseFloat(this.value);
        v = (typeof v === 'number' && !isNaN(v) && v >= 0) ? v : 0;
        let dwell = Math.max(SCHED_DWELL_FLOOR_MIN, v);
        let minDwell = f.minDwellMin != null ? f.minDwellMin : dwell;
        minDwell = Math.max(SCHED_DWELL_FLOOR_MIN, minDwell);
        if (minDwell > dwell) minDwell = dwell;
        f.dwellMin = dwell;
        f.minDwellMin = minDwell;
        this.value = f.dwellMin;
        if (minDwellEl) minDwellEl.value = f.minDwellMin;
        const tArr = f.timeMin != null ? f.timeMin : 0;
        f.sobtMin_orig = tArr + dwell;
        f.stotMin_orig = scheduledStotFromSobtMinutes(f, f.sobtMin_orig);
        f.sldtMin_orig = scheduledSldtFromSibtMinutes(f, tArr);
        if (typeof computeScheduledDisplayTimesIncremental === 'function') {
          const touched = f.standId ? [f.standId] : [];
          computeScheduledDisplayTimesIncremental(state.flights, new Set([f.id]), new Set(touched));
        }
        formatFlightScheduleGateInputs(f);
        if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
        if (typeof renderFlightList === 'function') {
          const rsD = f.standId ? [f.standId] : [];
          renderFlightList(false, false, { scheduleMode: 'incremental', dirtyFlightIds: [f.id], touchedStandIds: rsD });
        }
        rebuildSelectedFlightTimeline();
      });
    }
    if (minDwellEl) {
      minDwellEl.addEventListener('change', function() {
        if (!state.selectedObject || state.selectedObject.type !== 'flight') return;
        const f = state.selectedObject.obj;
        let dwell = f.dwellMin != null ? f.dwellMin : 0;
        dwell = Math.max(SCHED_DWELL_FLOOR_MIN, dwell);
        let v = parseFloat(this.value);
        v = (typeof v === 'number' && !isNaN(v) && v >= 0) ? v : 0;
        let minDwell = Math.max(SCHED_DWELL_FLOOR_MIN, v);
        if (minDwell > dwell) minDwell = dwell;
        f.dwellMin = dwell;
        f.minDwellMin = minDwell;
        if (dwellEl) dwellEl.value = f.dwellMin;
        this.value = f.minDwellMin;
        if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
        var rs4 = f.standId || null;
        if (typeof renderFlightList === 'function')
          renderFlightList(false, false, { scheduleMode: 'incremental', dirtyFlightIds: [f.id], touchedStandIds: rs4 ? [rs4] : [] });
      });
    }
    if (playBtn) {
      playBtn.addEventListener('click', function() {
        const errs = validateNetworkForFlights();
        if (errs.length) {
          state.simPlaying = false;
          updateFlightError(errs);
          alert('Simulation cannot be played:\\n' + errs.join('\\n'));
          return;
        }
        if (!state.flights.length) {
          updateFlightError('registered FlightThere is no.');
          alert('registered FlightThere is no.');
          return;
        }
        if (!state.globalUpdateFresh) {
          alert('Pro Sim(새로고침)이 필요합니다. 빨간 동기화 표시일 때는 타임라인이 비어 있어 재생할 수 없습니다.');
          return;
        }
        if (typeof recomputeSimDuration === 'function') recomputeSimDuration();
        const lo = state.simStartSec, hi = state.simDurationSec;
        let t = snapSimTimeSecForSlider(Math.max(lo, Math.min(hi, state.simTimeSec)));
        if (hi > lo && t >= hi - 1e-3) t = snapSimTimeSecForSlider(lo);
        state.simTimeSec = t;
        if (simSlider) simSlider.value = state.simTimeSec;
        state.simSliderScrubbing = false;
        if (typeof updateFlightSimPlaybackLabelsDom === 'function') updateFlightSimPlaybackLabelsDom();
        if (typeof prepareLazyTimelinesForCurrentSim === 'function') prepareLazyTimelinesForCurrentSim(state.simTimeSec);
        state.simPlaying = true;
        ensureSimLoop._lastTs = null;
        ensureSimLoop._playKick = true;
        ensureSimLoop();
        try { draw(); } catch(e) {}
        if (typeof update3DScene === 'function') update3DScene();
      });
    }
    if (pauseBtn) {
      pauseBtn.addEventListener('click', function() {
        state.simPlaying = false;
        if (typeof ensureSimLoop === 'function') ensureSimLoop._playKick = false;
      });
    }
    if (resetBtn) {
      resetBtn.addEventListener('click', function() {
        state.simPlaying = false;
        if (typeof ensureSimLoop === 'function') ensureSimLoop._playKick = false;
        state.simTimeSec = snapSimTimeSecForSlider(state.simStartSec);
        if (simSlider) simSlider.value = state.simTimeSec;
        if (typeof updateFlightSimPlaybackLabelsDom === 'function') updateFlightSimPlaybackLabelsDom();
        try { draw(); } catch(e) {}
        if (typeof update3DScene === 'function') update3DScene();
      });
    }
    let simSliderPointerActive = false;
    function finalizeSimSliderPointerDrag() {
      if (!simSliderPointerActive) return;
      simSliderPointerActive = false;
      state.simSliderScrubbing = false;
      if (typeof prepareLazyTimelinesForCurrentSim === 'function') prepareLazyTimelinesForCurrentSim(state.simTimeSec);
      if (typeof updateFlightSimPlaybackLabelsDom === 'function') updateFlightSimPlaybackLabelsDom();
      try { draw(); } catch(e) {}
      if (typeof update3DScene === 'function') update3DScene();
    }
    if (simSlider) {
      simSlider.addEventListener('pointerdown', function(e) {
        if (e.button != null && e.button !== 0) return;
        if (e.isPrimary === false) return;
        simSliderPointerActive = true;
        state.simSliderScrubbing = true;
        try { simSlider.setPointerCapture(e.pointerId); } catch (err) {}
      });
      simSlider.addEventListener('pointerup', function(e) {
        if (!simSliderPointerActive) return;
        try { simSlider.releasePointerCapture(e.pointerId); } catch (err2) {}
        finalizeSimSliderPointerDrag();
      });
      simSlider.addEventListener('pointercancel', function() {
        finalizeSimSliderPointerDrag();
      });
      simSlider.addEventListener('lostpointercapture', function() {
        finalizeSimSliderPointerDrag();
      });
      simSlider.addEventListener('input', function() {
        const secs = parseFloat(this.value);
        if (!isNaN(secs)) {
          const snapped = snapSimTimeSecForSlider(secs);
          state.simTimeSec = snapped;
          this.value = snapped;
          if (typeof updateFlightSimPlaybackLabelsDom === 'function') updateFlightSimPlaybackLabelsDom();
          if (state.simSliderScrubbing) return;
          if (typeof prepareLazyTimelinesForCurrentSim === 'function') prepareLazyTimelinesForCurrentSim(state.simTimeSec);
          try { draw(); } catch(e) {}
          if (typeof update3DScene === 'function') update3DScene();
        }
      });
    }
    if (speedSelect) {
      speedSelect.addEventListener('change', function() {
        const v = parseFloat(this.value);
        state.simSpeed = !isNaN(v) && v > 0 ? v : 1;
      });
      const v0 = parseFloat(speedSelect.value);
      state.simSpeed = !isNaN(v0) && v0 > 0 ? v0 : _dc.defaultSimSpeed;
    }
    const btnHideSimBar = document.getElementById('btnHideSimPlaybackBar');
    if (btnHideSimBar) {
      btnHideSimBar.addEventListener('click', function() {
        state.simPlaybackDockVisible = false;
        if (typeof applySimPlaybackBarDomVisibility === 'function') applySimPlaybackBarDomVisibility();
      });
    }
    function syncTableToFlightState() {
      const schedTable = document.querySelector('.flight-schedule-table');
      if (!schedTable || !Array.isArray(state.flights)) return;
      const rows = Array.from(schedTable.querySelectorAll('tbody tr.flight-data-row'));
      rows.forEach(function(row) {
        const fid = row.getAttribute('data-id');
        if (!fid) return;
        const f = state.flights.find(function(ff) { return ff && ff.id === fid; });
        if (!f) return;
        const tds = Array.from(row.querySelectorAll('td'));
        if (tds.length <= FLIGHT_SCHED_TD_ETOT) return;
        const getMin = function(idx) {
          const td = tds[idx];
          if (!td) return null;
          const dm = td.getAttribute('data-sched-min');
          if (dm != null && String(dm).trim() !== '') {
            const n = parseFloat(dm);
            return isFinite(n) ? n : null;
          }
          const txt = (td.textContent || '').trim();
          if (!txt) return null;
          const parsed = parseTimeToMinutes(txt);
          return isFinite(parsed) ? parsed : null;
        };
        const map = {
          sldtMin_d: FLIGHT_SCHED_TD_SLD,
          sibtMin_d: FLIGHT_SCHED_TD_SIBTD,
          sobtMin_d: FLIGHT_SCHED_TD_SOBTD,
          stotMin_d: FLIGHT_SCHED_TD_STOTD,
          eldtMin: FLIGHT_SCHED_TD_ELDT,
          eibtMin: FLIGHT_SCHED_TD_EIBT,
          eobtMin: FLIGHT_SCHED_TD_EOBT,
          etotMin: FLIGHT_SCHED_TD_ETOT
        };
        Object.keys(map).forEach(function(key) {
          const v = getMin(map[key]);
          if (v != null) f[key] = v;
        });
      });
    }
    function setLayoutMessage(msg, isError) {
      if (!layoutMsgEl) return;
      layoutMsgEl.textContent = msg || '';
      layoutMsgEl.style.color = isError ? '#f97316' : '#9ca3af';
    }
    if (saveLayoutBtn) {
      saveLayoutBtn.addEventListener('click', function() {
        const name = (layoutNameInput && layoutNameInput.value || '').trim();
        if (!name) {
          setLayoutMessage('Please enter a save name.', true);
          return;
        }
        try {
          if (typeof syncStateFromPanel === 'function') syncStateFromPanel();
          if (typeof syncTableToFlightState === 'function') syncTableToFlightState();
          const data = serializeCurrentLayout();
          fetchSaveLayout(name, data).then(function(r) {
            if (r.ok) {
              if (typeof updateLayoutNameBar === 'function') updateLayoutNameBar(name);
              setLayoutMessage('Saved to Layout_storage as "' + name + '.json"', false);
            } else setLayoutMessage('save failed (status ' + r.status + ') — python run_app.pyAfter running with http://127.0.0.1:8501 connection', true);
          }).catch(function(e) {
            console.warn('Layout save fetch failed', e);
            setLayoutMessage('Connection failed: ' + (e && e.message) + ' — python run_app.pyAfter running with http://127.0.0.1:8501 connection', true);
          });
        } catch (e) {
          console.error(e);
          setLayoutMessage('Unable to save layout.', true);
        }
      });
    }
    function switchLayoutTab(tabId) {
      const root = document.getElementById('tab-saveload');
      if (!root) return;
      root.querySelectorAll('.layout-save-load-tab').forEach(btn => btn.classList.remove('active'));
      root.querySelectorAll('.layout-save-load-pane').forEach(p => p.classList.remove('active'));
      const btn = root.querySelector('.layout-save-load-tab[data-sltab="' + tabId + '"]');
      const pane = document.getElementById('layout-' + tabId + '-pane');
      if (btn) btn.classList.add('active');
      if (pane) pane.classList.add('active');
      if (tabId === 'load') fetchAndRefreshLayoutList();
    }
    const layoutMessageSaveEl = document.getElementById('layoutMessageSave');
    const btnSaveCurrent = document.getElementById('btnSaveCurrentLayout');
    if (btnSaveCurrent) btnSaveCurrent.addEventListener('click', function() {
      const name = (state.currentLayoutName && state.currentLayoutName.trim()) || (INITIAL_LAYOUT_DISPLAY_NAME || 'default_layout');
      try {
        if (typeof syncStateFromPanel === 'function') syncStateFromPanel();
        if (typeof syncTableToFlightState === 'function') syncTableToFlightState();
        const data = serializeCurrentLayout();
        fetchSaveLayout(name, data).then(function(r) {
          if (r.ok) {
            if (layoutMessageSaveEl) { layoutMessageSaveEl.textContent = 'saved: ' + name + '.json'; layoutMessageSaveEl.style.color = '#9ca3af'; }
          } else if (layoutMessageSaveEl) { layoutMessageSaveEl.textContent = 'save failed (status ' + r.status + ')'; layoutMessageSaveEl.style.color = '#f97316'; }
        }).catch(function(e) {
          console.warn('Object save fetch failed', e);
          if (layoutMessageSaveEl) { layoutMessageSaveEl.textContent = 'Connection failed: ' + (e && e.message); layoutMessageSaveEl.style.color = '#f97316'; }
        });
      } catch (e) { if (layoutMessageSaveEl) { layoutMessageSaveEl.textContent = 'error: ' + (e && e.message); layoutMessageSaveEl.style.color = '#f97316'; } }
    });
    const saveLoadTabRoot = document.getElementById('tab-saveload');
    if (saveLoadTabRoot) {
      saveLoadTabRoot.querySelectorAll('.layout-save-load-tab[data-sltab]').forEach(btn => {
        btn.addEventListener('click', function() { switchLayoutTab(this.getAttribute('data-sltab')); });
      });
    }
    function getLayoutApiBase() {
      if (LAYOUT_API_URL && LAYOUT_API_URL !== 'null') return LAYOUT_API_URL;
      try { if (window.location && window.location.origin && window.location.origin !== 'null') return window.location.origin; } catch(e) {}
      return '';
    }
    function fetchSaveLayout(name, data) {
      const apiBase = (typeof getLayoutApiBase === 'function') ? getLayoutApiBase() : (LAYOUT_API_URL || '');
      return fetch(apiBase + '/api/save-layout', {
        method: 'POST',
