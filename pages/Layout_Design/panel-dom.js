    const opts = AIRCRAFT_TYPES.map(a => '<option value="' + escapeHtml(String(a.id || a.name || '')) + '">' + escapeHtml(a.name || a.id || '') + '</option>').join('');
    sel.innerHTML = opts || '<option value="A320">Airbus A320</option>';
    if (!opts && sel.options.length) sel.value = 'A320';
    else if (sel.options.length) sel.value = sel.options[0].value;
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
    const aircraftEl = document.getElementById('flightAircraftType');
    const regEl = document.getElementById('flightReg');
    const layoutNameInput = document.getElementById('layoutName');
    const saveLayoutBtn = document.getElementById('btnSaveLayout');
    const layoutMsgEl = document.getElementById('layoutMessage');
    const layoutLoadListEl = document.getElementById('layoutLoadList');
    const globalUpdateBtn = document.getElementById('btnGlobalUpdate');
    if (!arrDepEl) return;
    populateAircraftSelect(aircraftEl);

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
    if (globalUpdateBtn) {
      globalUpdateBtn.addEventListener('click', function() {
        function failGlobalUpdate(err) {
          console.error('Global update error', err);
          if (typeof setGlobalUpdateProgressUi === 'function') setGlobalUpdateProgressUi(false);
        }
        if (typeof setGlobalUpdateProgressUi === 'function')
          setGlobalUpdateProgressUi(true, '동기화 중…', 5);
        scheduleAfterPaint(function globalUpdateStep1() {
          try {
            if (typeof syncPanelFromState === 'function') syncPanelFromState();
            if (typeof setGlobalUpdateProgressUi === 'function')
              setGlobalUpdateProgressUi(true, '항공 경로·타임라인…', 22);
          } catch (e) { failGlobalUpdate(e); return; }
          setTimeout(function globalUpdateStep2() {
            try {
              function runAfterFlightListRefresh() {
                try {
                  if (typeof setGlobalUpdateProgressUi === 'function')
                    setGlobalUpdateProgressUi(true, 'KPI·캔버스…', 92);
                } catch (e2) { failGlobalUpdate(e2); return; }
                setTimeout(function globalUpdateStep6() {
                  try {
                    if (typeof renderKpiDashboard === 'function') renderKpiDashboard('Updated');
                    if (typeof syncSimulationPlaybackAfterTimelines === 'function') syncSimulationPlaybackAfterTimelines();
                    if (typeof markGlobalUpdateFresh === 'function') markGlobalUpdateFresh();
                    if (typeof draw === 'function') draw();
                    if (typeof update3DScene === 'function') update3DScene();
                  } catch (e3) { failGlobalUpdate(e3); return; }
                  if (typeof setGlobalUpdateProgressUi === 'function') setGlobalUpdateProgressUi(false);
                }, 0);
              }
              function runFlightListThenKpi() {
                setTimeout(function globalUpdateStep5() {
                  try {
                    if (typeof renderFlightList === 'function')
                      renderFlightList(false, true, undefined, runAfterFlightListRefresh);
                    else
                      runAfterFlightListRefresh();
                  } catch (e2) { failGlobalUpdate(e2); return; }
                }, 0);
              }
              function runSchedAndRwyPanels() {
                setTimeout(function globalUpdateStep3() {
                  try {
                    if (typeof bumpVttArrCacheRev === 'function') bumpVttArrCacheRev();
                    if (typeof computeScheduledDisplayTimes === 'function') computeScheduledDisplayTimes(state.flights);
                    if (typeof computeSeparationAdjustedTimes === 'function') computeSeparationAdjustedTimes();
                    if (typeof syncSimulationPlaybackAfterTimelines === 'function') syncSimulationPlaybackAfterTimelines();
                    if (typeof setGlobalUpdateProgressUi === 'function')
                      setGlobalUpdateProgressUi(true, 'Runway 패널…', 62);
                  } catch (e2) { failGlobalUpdate(e2); return; }
                  setTimeout(function globalUpdateStep4() {
                    try {
                      if (typeof renderRunwaySeparation === 'function') renderRunwaySeparation();
                      if (typeof setGlobalUpdateProgressUi === 'function')
                        setGlobalUpdateProgressUi(true, '항공편 표·간트…', 78);
                    } catch (e3) { failGlobalUpdate(e3); return; }
                    runFlightListThenKpi();
                  }, 0);
                }, 0);
              }
              if (typeof updateAllFlightPaths === 'function') {
                updateAllFlightPaths(function globalUpdatePathsDone() {
                  try {
                    if (typeof setGlobalUpdateProgressUi === 'function')
                      setGlobalUpdateProgressUi(true, 'RET·스케줄·활주로 분리…', 48);
                  } catch (e2) { failGlobalUpdate(e2); return; }
                  runSchedAndRwyPanels();
                });
              } else {
                if (typeof recomputeSimDuration === 'function') recomputeSimDuration();
                if (typeof setGlobalUpdateProgressUi === 'function')
                  setGlobalUpdateProgressUi(true, 'RET·스케줄·활주로 분리…', 48);
                runSchedAndRwyPanels();
              }
            } catch (e) { failGlobalUpdate(e); return; }
          }, 0);
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
          const defMin = getDefaultSibtMinutes();
          timeStr = formatMinutesToHHMMSS(defMin);
          if (timeInputEl) timeInputEl.value = timeStr;
        }
        const timeMin = parseTimeToMinutes(timeStr);
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
        f.deferPathCompute = true;
        state.flights.push(f);
        if (typeof syncSimulationPlaybackAfterTimelines === 'function') syncSimulationPlaybackAfterTimelines();
        else if (typeof recomputeSimDuration === 'function') recomputeSimDuration();
        if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
        var addTouched = f.standId ? [f.standId] : [];
        renderFlightList(false, false, { scheduleMode: 'incremental', dirtyFlightIds: [f.id], touchedStandIds: addTouched });
        if (timeInputEl) {
          const nextDef = getDefaultSibtMinutes();
          timeInputEl.value = formatMinutesToHHMMSS(nextDef);
        }
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
      if (timeInputEl) timeInputEl.value = formatMinutesToHHMMSS(f.timeMin);
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
      computeFlightPath(f, 'arrival');
      computeFlightPath(f, 'departure');
      const isArr = f.arrDep !== 'Dep';
      if (isArr && arrivalAirsideBlocked(f)) {
        updateFlightError(f.arrRetFailed && !f.noWayArr ? 'no path(No Way): Arrival RET failed.' : 'no path(No Way): Arrival route not found.');
        f.timeline = null;
        draw();
        return;
      }
      if (!isArr && f.noWayDep) {
        updateFlightError('no path(No Way): Departure route not found.');
        f.timeline = null;
        draw();
        return;
      }
      if (typeof buildFullAirsideTimelineForFlight === 'function') buildFullAirsideTimelineForFlight(f);
      if (!f.timeline || !f.timeline.length) {
        updateFlightError('No valid route found on that network. (After changing settings)');
        return;
      }
      if (typeof syncSimulationPlaybackAfterTimelines === 'function') syncSimulationPlaybackAfterTimelines();
      else if (typeof recomputeSimDuration === 'function') recomputeSimDuration();
      var sidSched = f.standId || null;
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
    if (timeInputEl) {
      timeInputEl.addEventListener('change', function() {
        if (!state.selectedObject || state.selectedObject.type !== 'flight') return;
        const f = state.selectedObject.obj;
        const mins = parseTimeToMinutes(this.value || '0');
        f.timeMin = mins;
        this.value = formatMinutesToHHMMSS(mins);
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
          alert('Light Sim(새로고침)이 필요합니다. 빨간 동기화 표시일 때는 타임라인이 비어 있어 재생할 수 없습니다.');
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
        if (tds.length < 15) return;
        const getMin = function(idx) {
          const txt = (tds[idx] && tds[idx].textContent || '').trim();
          if (!txt) return null;
          const parts = txt.split(':');
          if (parts.length >= 2) {
            const h = parseInt(parts[0], 10) || 0;
            const m = parseInt(parts[1], 10) || 0;
            const s = parts.length >= 3 ? (parseInt(parts[2], 10) || 0) : 0;
            return h * 60 + m + s / 60;
          }
          const n = parseFloat(txt);
          return isNaN(n) ? null : n;
        };
        const map = {
          sldtMin_d: 7, sibtMin_d: 8, sobtMin_d: 9,  stotMin_d: 10,
          eldtMin:  11, eibtMin:  12, eobtMin:  13, etotMin:   14
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
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ layout: data, name: name })
      });
    }
    function fetchAndRefreshLayoutList() {
      if (!layoutLoadListEl) return;
      layoutLoadListEl.innerHTML = '<div style="font-size:11px;color:#9ca3af;">Loading list...</div>';
      const apiBase = getLayoutApiBase();
      fetch(apiBase + '/api/list-layouts').then(function(r) {
        if (!r.ok) throw new Error('API Connection failed (status ' + r.status + ')');
        return r.json();
      }).then(function(data) {
        const names = (data && data.names) ? data.names : (Array.isArray(LAYOUT_NAMES) ? LAYOUT_NAMES : []);
        refreshLayoutLoadList(names);
      }).catch(function(e) {
        console.warn('Layout list fetch failed', e);
        layoutLoadListEl.innerHTML = '<div style="font-size:11px;color:#f97316;">Connection failed: ' + (e && e.message) + '</div><div style="font-size:10px;color:#9ca3af;margin-top:4px;">python run_app.py After running with http://127.0.0.1:8501 connection</div>';
      });
    }
    function refreshLayoutLoadList(namesFromApi) {
      if (!layoutLoadListEl) return;
      const names = namesFromApi != null ? (Array.isArray(namesFromApi) ? namesFromApi : []) : (Array.isArray(LAYOUT_NAMES) ? LAYOUT_NAMES : []);
      if (!names.length) {
        layoutLoadListEl.innerHTML = '<div style="font-size:11px;color:#9ca3af;">There are no saved layouts.</div>';
        return;
      }
      const reserved = { 'default_layout': true, 'current_layout': true };
      layoutLoadListEl.innerHTML = names.map(function(name) {
        const n = (name || '').replace(/"/g, '&quot;').replace(/</g, '&lt;');
        const showDel = !reserved[(name || '').toLowerCase()];
        const delBtn = showDel ? '<button type="button" class="layout-load-delete" title="Delete" data-name="' + (name || '').replace(/"/g, '&quot;') + '">×</button>' : '';
        return '<div class="layout-load-item" data-name="' + (name || '').replace(/"/g, '&quot;') + '"><span class="layout-load-name">' + n + '</span>' + delBtn + '</div>';
      }).join('');
      layoutLoadListEl.querySelectorAll('.layout-load-item').forEach(function(el) {
        const name = el.getAttribute('data-name');
        el.addEventListener('click', function(ev) {
          if (ev.target && ev.target.classList && ev.target.classList.contains('layout-load-delete')) return;
          if (!name) return;
          var apiBase = getLayoutApiBase();
          if (layoutMsgEl) { layoutMsgEl.textContent = 'Loading...'; layoutMsgEl.style.color = '#9ca3af'; }
          fetch(apiBase + '/api/load-layout?name=' + encodeURIComponent(name)).then(function(r) {
            if (!r.ok) throw new Error('not_found');
            return r.json();
          }).then(function(obj) {
            if (!obj || typeof obj !== 'object') { throw new Error('invalid_response'); }
            try {
              state.hasSimulationResult = false;
              applyLayoutObject(obj);
              resizeCanvas();
              reset2DView();
              syncPanelFromState();
              if (typeof draw === 'function') draw();
              if (typeof update3DScene === 'function') update3DScene();
              if (typeof updateLayoutNameBar === 'function') updateLayoutNameBar(name);
              if (typeof recomputeSimDuration === 'function') recomputeSimDuration();
              if (layoutMsgEl) { layoutMsgEl.textContent = 'Loaded \"' + name + '\"'; layoutMsgEl.style.color = '#9ca3af'; }
            } catch (err) {
              console.error('applyLayoutObject error', err);
              throw err;
            }
          }).catch(function(e) {
            console.warn('Layout load fetch failed', e);
            if (layoutMsgEl) { layoutMsgEl.textContent = 'Failed to load: ' + ((e && e.message) || name || '') + ' — python run_app.pyAfter running with http://127.0.0.1:8501 connection'; layoutMsgEl.style.color = '#f97316'; }
          });
        });
        el.querySelector('.layout-load-delete') && el.querySelector('.layout-load-delete').addEventListener('click', function(ev) {
          ev.stopPropagation();
          const n = this.getAttribute('data-name');
          if (!n) return;
          const apiBase = getLayoutApiBase();
          fetch(apiBase + '/api/delete-layout', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: n })
          }).then(function(r) {
            if (!r.ok) return r.json().then(function(d) { throw new Error(d.error || 'Deletion failed'); });
            return fetch(apiBase + '/api/list-layouts').then(function(r2) { return r2.json(); });
          }).then(function(data) {
            if (data && data.names) refreshLayoutLoadList(data.names);
            if (layoutMsgEl) { layoutMsgEl.textContent = 'deleted.'; layoutMsgEl.style.color = '#9ca3af'; }
          }).catch(function(e) {
            console.warn('Layout delete fetch failed', e);
            if (layoutMsgEl) { layoutMsgEl.textContent = ((e && e.message) || 'Deletion failed') + ' — python run_app.pyAfter running with http://127.0.0.1:8501 connection'; layoutMsgEl.style.color = '#f97316'; }
          });
        });
      });
    }
    fetch((getLayoutApiBase() || '') + '/api/list-layouts').then(function(r) {
      if (r.ok) return;
      var banner = document.getElementById('api-warning-banner');
      if (banner) banner.style.display = 'block';
    }).catch(function(e) {
      console.warn('API health check failed', e);
      var banner = document.getElementById('api-warning-banner');
      if (banner) banner.style.display = 'block';
    });
  })();

  document.getElementById('btnTerminalDraw').addEventListener('click', function() {
    state.selectedObject = null;
    if (state.terminalDrawingId) {
      const t = state.terminals.find(x => x.id === state.terminalDrawingId);
      if (t && !t.closed && t.vertices.length >= 3) {
        t.closed = true;
        if (terminalOverlapsAnyTaxiway(t)) {
          alert('this Apron/Terminalsilver Taxiway Overlaps the center line. Please place it in a different location.');
          state.terminals = state.terminals.filter(term => term.id !== t.id);
        }
      }
      state.terminalDrawingId = null;
      state.layoutPathDrawPointer = null;
      syncPanelFromState();
      draw();
      return;
    }
    const selectedBuildingType = normalizeBuildingType(document.getElementById('buildingType') ? document.getElementById('buildingType').value : BUILDING_TYPE_DEFAULT);
    const nameBase = document.getElementById('terminalName').value.trim() || getDefaultBuildingNameForType(selectedBuildingType);
    const floorsEl = document.getElementById('terminalFloors');
    const f2fEl = document.getElementById('terminalFloorToFloor');
    let floors = floorsEl ? parseInt(floorsEl.value, 10) : 1;
    let f2f = f2fEl ? Number(f2fEl.value) : 4;
    floors = Math.max(1, floors || 1);
    f2f = Math.max(0.5, f2f || 4);
    const totalH = floors * f2f;
    if (findDuplicateLayoutName('terminal', null, nameBase)) {
      alertDuplicateLayoutName();
      return;
    }
    const term = { id: id(), name: nameBase, buildingType: selectedBuildingType, vertices: [], closed: false, floors, floorToFloor: f2f, floorHeight: totalH, departureCapacity: 0, arrivalCapacity: 0 };
    pushUndo();
    state.terminals.push(term);
    state.currentTerminalId = term.id;
    state.terminalDrawingId = term.id;
    syncPanelFromState();
    draw();
    if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
  });

  document.getElementById('btnTaxiwayDraw').addEventListener('click', function() {
    const hadSelection = !!state.selectedObject;
    state.selectedObject = null;
    if (state.taxiwayDrawingId) {
      const tw = state.taxiways.find(x => x.id === state.taxiwayDrawingId);
      if (tw && tw.vertices.length >= 2) {
        if (taxiwayOverlapsAnyTerminal(tw)) {
          alert('this TaxiwayIs TerminalIt overlaps with . Please draw a different path.');
          pushUndo();
          state.taxiways = state.taxiways.filter(t => t.id !== tw.id);
        }
        state.taxiwayDrawingId = null;
        state.layoutPathDrawPointer = null;
        syncPanelFromState();
        if (typeof redrawLayoutAfterEdit === 'function') redrawLayoutAfterEdit();
        else if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths(); else draw();
      }
      return;
    }
    const layoutMode = settingModeSelect ? settingModeSelect.value : 'taxiway';
    const pathType = pathTypeFromLayoutMode(isPathLayoutMode(layoutMode) ? layoutMode : 'taxiway');
    const nameInputEl = document.getElementById('taxiwayName');
    const defaultPathName = getDefaultPathName(pathType);
    if (hadSelection && nameInputEl) nameInputEl.value = defaultPathName;
    const rawName = nameInputEl ? nameInputEl.value.trim() : '';
    const nameBase = rawName || defaultPathName;
    const inputWidth = Number(document.getElementById('taxiwayWidth').value);
    const baseWidth = pathType === 'runway'
      ? RUNWAY_PATH_DEFAULT_WIDTH
      : (pathType === 'runway_exit' ? RUNWAY_EXIT_DEFAULT_WIDTH : TAXIWAY_DEFAULT_WIDTH);
    const widthVal = clampTaxiwayWidthM(pathType, inputWidth, baseWidth);
    const modeVal = (function() {
      const raw = document.getElementById('taxiwayDirectionMode') ? document.getElementById('taxiwayDirectionMode').value : '';
      if (pathType === 'runway') return (raw === 'counter_clockwise') ? 'counter_clockwise' : 'clockwise';
      return raw || 'both';
    })();
    const maxExitInput = document.getElementById('taxiwayMaxExitVel');
    const minExitInput = document.getElementById('taxiwayMinExitVel');
    const maxExitVelocity = (pathType === 'runway_exit' && maxExitInput)
      ? (function() { const mv = Number(maxExitInput.value); return isFinite(mv) && mv > 0 ? mv : null; })()
      : null;
    const minExitVelocity = (pathType === 'runway_exit' && minExitInput)
      ? (function() {
          const mv = Number(minExitInput.value);
          if (!isFinite(mv) || mv <= 0) return 15;
          if (maxExitVelocity != null && mv > maxExitVelocity) return maxExitVelocity;
          return mv;
        })()
      : undefined;
    const allowedRwDirections = (pathType === 'runway_exit')
      ? getRunwayExitAllowedDirectionsFromPanel()
      : undefined;
    const minArrVelInput = document.getElementById('runwayMinArrVelocity');
    const minArrVelocity = (pathType === 'runway' && minArrVelInput)
      ? (function() {
          const mv = Number(minArrVelInput.value);
          return (isFinite(mv) && mv > 0) ? Math.max(1, Math.min(150, mv)) : 15;
        })()
      : undefined;
    const lineupEl = document.getElementById('runwayLineupDistM');
    const lineupDistM = (pathType === 'runway' && lineupEl)
      ? (function() { const x = Number(lineupEl.value); return (isFinite(x) && x >= 0) ? x : 0; })()
      : undefined;
    const runwayStartDispEl = document.getElementById('runwayStartDisplacedThresholdM');
    const startDisplacedThresholdM = (pathType === 'runway' && runwayStartDispEl)
      ? (function() { const x = Number(runwayStartDispEl.value); return (isFinite(x) && x >= 0) ? x : RUNWAY_START_DISPLACED_THRESHOLD_DEFAULT_M; })()
      : undefined;
    const runwayStartBlastEl = document.getElementById('runwayStartBlastPadM');
    const startBlastPadM = (pathType === 'runway' && runwayStartBlastEl)
      ? (function() { const x = Number(runwayStartBlastEl.value); return (isFinite(x) && x >= 0) ? x : RUNWAY_START_BLAST_PAD_DEFAULT_M; })()
      : undefined;
    const runwayEndDispEl = document.getElementById('runwayEndDisplacedThresholdM');
    const endDisplacedThresholdM = (pathType === 'runway' && runwayEndDispEl)
      ? (function() { const x = Number(runwayEndDispEl.value); return (isFinite(x) && x >= 0) ? x : RUNWAY_END_DISPLACED_THRESHOLD_DEFAULT_M; })()
      : undefined;
    const runwayEndBlastEl = document.getElementById('runwayEndBlastPadM');
    const endBlastPadM = (pathType === 'runway' && runwayEndBlastEl)
      ? (function() { const x = Number(runwayEndBlastEl.value); return (isFinite(x) && x >= 0) ? x : RUNWAY_END_BLAST_PAD_DEFAULT_M; })()
      : undefined;
    const taxiway = { id: id(), name: nameBase, vertices: [], width: widthVal, direction: modeVal, pathType, maxExitVelocity, minExitVelocity, allowedRwDirections, minArrVelocity, lineupDistM, avgMoveVelocity: (function() {
      const el = document.getElementById('taxiwayAvgMoveVelocity');
      const v = el ? Number(el.value) : 10;
      return (typeof v === 'number' && isFinite(v) && v > 0) ? Math.max(1, Math.min(50, v)) : 10;
    })(), startDisplacedThresholdM, startBlastPadM, endDisplacedThresholdM, endBlastPadM };
    if (pathType !== 'runway') delete taxiway.minArrVelocity;
    if (pathType !== 'runway') delete taxiway.lineupDistM;
    if (pathType !== 'runway') delete taxiway.startDisplacedThresholdM;
    if (pathType !== 'runway') delete taxiway.startBlastPadM;
    if (pathType !== 'runway') delete taxiway.endDisplacedThresholdM;
    if (pathType !== 'runway') delete taxiway.endBlastPadM;
    if (pathType !== 'runway_exit') { delete taxiway.maxExitVelocity; delete taxiway.minExitVelocity; delete taxiway.allowedRwDirections; }
    if (findDuplicateLayoutName('taxiway', null, nameBase)) {
      alertDuplicateLayoutName();
      return;
    }
    pushUndo();
    state.taxiways.push(taxiway);
    state.taxiwayDrawingId = taxiway.id;
    syncPanelFromState();
    if (typeof redrawLayoutAfterEdit === 'function') redrawLayoutAfterEdit();
    else if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths(); else draw();
  });
  const btnPbbDrawEl = document.getElementById('btnPbbDraw');
  if (btnPbbDrawEl) btnPbbDrawEl.addEventListener('click', function() {
    toggleLayoutDrawMode('pbbDrawing', 'previewPbb', null);
  });
  const btnRemoteDrawEl = document.getElementById('btnRemoteDraw');
