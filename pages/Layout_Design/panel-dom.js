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
    if (!forceResample && f.timeline_meta && typeof f.timeline_meta === 'object' &&
        f.timeline_meta.playbackSource === 'des_result') {
      f.__schedRetRotRev = rev;
      return;
    }
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
    const sibt = f.sibtMin != null ? f.sibtMin : (f.timeMin != null ? f.timeMin : 0);
    const dwell = f.dwellMin != null ? f.dwellMin : 0;
    const sobt = f.sobtMin != null ? f.sobtMin : (sibt + dwell);
    return {
      sldt: f.sldtMin != null ? f.sldtMin : Math.max(0, sibt - SCHED_SIBT_MINUS_SLDT_MIN),
      sibt: sibt,
      sobt: sobt,
      stot: f.stotMin != null ? f.stotMin : (sobt + SCHED_STOT_MINUS_SOBT_MIN),
    };
  }

  function flightScheduleProSimTimedCell(displayStr, dotKind) {
    const d = '—';
    const has = displayStr != null && String(displayStr).trim() !== '' && displayStr !== d;
    const text = has ? String(displayStr) : d;
    const muted = has ? '' : ' flight-sched-dot--muted';
    let dotClass = 'flight-sched-dot--green';
    if (dotKind === 'vttarr') dotClass = 'flight-sched-dot--vttarr';
    else if (dotKind === 'dttarr') dotClass = 'flight-sched-dot--dttarr';
    else if (dotKind === 'dttdep') dotClass = 'flight-sched-dot--dttdep';
    else if (dotKind === 'pushback') dotClass = 'flight-sched-dot--pushback';
    else if (dotKind === 'red') dotClass = 'flight-sched-dot--red';
    else if (dotKind === 'pink') dotClass = 'flight-sched-dot--pink';
    return '<span class="flight-sched-cell-inner">' +
      '<span class="flight-sched-dot ' + dotClass + muted + '" aria-hidden="true"></span>' +
      '<span class="flight-sched-cell-text">' + (has ? escapeHtml(text) : d) + '</span></span>';
  }

  function _buildFlightListHeaderHtml(apronK) {
    const k = Math.max(1, Number(apronK) || flightScheduleColumnK());
    const sHeads = [];
    const eHeads = [];
    const apHeads = [];
    for (let i = 1; i <= k; i++) {
      sHeads.push('<th class="flight-col-s' + (i === 1 ? ' flight-col-s-start flight-td-sibt' : '') + '">SIBT' + i + '</th>');
      sHeads.push('<th class="flight-col-s' + (i === k ? ' flight-col-s-last' : '') + '">SOBT' + i + '</th>');
      eHeads.push('<th class="flight-col-e">EIBT' + i + '</th>');
      eHeads.push('<th class="flight-col-e">EOBT' + i + '</th>');
      apHeads.push('<th class="flight-th-mixed">AP' + i + '</th>');
    }
    return '' +
      '<table class="flight-schedule-table">' +
      '<thead><tr>' +
        '<th>Reg</th>' +
        '<th class="flight-th-mixed">Airline</th>' +
        '<th class="flight-th-mixed">Flight Num</th>' +
        '<th>ICAO Code</th>' +
        '<th class="flight-th-mixed">ICAO CAT</th>' +
        '<th>Int/Dom</th>' +
        '<th>Arr Rw</th>' +
        '<th>Arr RET</th>' +
        '<th>Arr Building</th>' +
        '<th>Dep Building</th>' +
        apHeads.join('') +
        '<th class="flight-th-mixed">Lookahead_taxi</th>' +
        '<th>Dep Rw</th>' +
        sHeads.join('') +
        '<th class="flight-col-e flight-col-e-start">ELDT</th>' +
        eHeads.join('') +
        '<th class="flight-col-e">ETOT</th>' +
        '<th class="flight-col-e flight-col-rot flight-th-mixed">ROT(arr)</th>' +
        '<th class="flight-th-mixed">VTT(Arr)</th>' +
        '<th class="flight-th-mixed">DTT(Arr)</th>' +
        '<th class="flight-th-mixed">PUSHBACK</th>' +
        '<th class="flight-th-mixed">DTT(Dep)</th>' +
        '<th class="flight-th-mixed">VTT(Dep)</th>' +
        '<th class="flight-col-e flight-col-rot flight-th-mixed">ROT(dep)</th>' +
        '<th class="flight-td-del"></th>' +
      '</tr></thead>' +
      '<tbody>';
  }

  function flightScheduleSegmentsForDisplay(f, apronK) {
    const raw = typeof normalizeFlightApronStaySegments === 'function' ? normalizeFlightApronStaySegments(f) : [];
    const merged = mergeAdjacentSameStandApronSegments(raw);
    const k = Math.max(1, Number(apronK) || flightScheduleColumnK());
    const out = [];
    for (let i = 0; i < k; i++) out.push(merged[i] || null);
    return out;
  }
  function flightScheduleStandLabelById(standId) {
    if (standId == null || standId === '') return '—';
    const st = typeof findStandById === 'function' ? findStandById(standId) : null;
    if (!st) return String(standId);
    return (st.name && String(st.name).trim()) || String(st.id || standId);
  }
  function _buildFlightListRowHtml(f, retStatsAll, apronK) {
    const k = Math.max(1, Number(apronK) || flightScheduleColumnK());
    const arrRunwayId = resolveArrivalRunwayIdForFlight(f);
    const ac = typeof getAircraftInfoByType === 'function' ? getAircraftInfoByType(f.aircraftType) : null;
    const arrRetFailed = isFlightCountedInArrivalConfigFailedRow(f, retStatsAll);
    let sampledRetName = '—';
    if (arrRetFailed) sampledRetName = 'Failed';
    else if (f.sampledArrRet != null && retStatsAll && retStatsAll.length) {
      const retInfo = retStatsAll.find(r => r.exit && r.exit.id === f.sampledArrRet);
      sampledRetName = retInfo ? (retInfo.name || 'RET') : 'RET';
    }
    const tArrMin = f.sibtMin != null ? f.sibtMin : (f.timeMin != null ? f.timeMin : 0);
    const dwell = f.dwellMin != null ? f.dwellMin : 0;
    const tDepMin = f.sobtMin != null ? f.sobtMin : (tArrMin + dwell);
    const schedDepRotMin = Math.max(0, Number(SCHED_DEP_ROT_MIN) || 2);
    const sldtCalc = (f.sldtMin != null ? f.sldtMin : Math.max(0, tArrMin));
    const stotCalc = (f.stotMin != null) ? f.stotMin : (tDepMin + schedDepRotMin);
    if (f.sibtMin == null || f.sobtMin == null || f.sldtMin == null || f.stotMin == null) {
      f.sldtMin = sldtCalc;
      f.sibtMin = tArrMin;
      f.sobtMin = tDepMin;
      f.stotMin = stotCalc;
    }
    const schedM = flightScheduleMinutesForRow(f);
    const sibtDisp = formatFlightScheduleDateTime(f, schedM.sibt);
    const sobtDisp = formatFlightScheduleDateTime(f, schedM.sobt);
    const segCells = flightScheduleSegmentsForDisplay(f, k);
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
    const dttArrStr = (f.proSimDttArrSec != null && isFinite(f.proSimDttArrSec)) ? formatSecondsToHHMMSS(f.proSimDttArrSec) : dash;
    const pushbackStr = (f.proSimPushbackSec != null && isFinite(f.proSimPushbackSec)) ? formatSecondsToHHMMSS(f.proSimPushbackSec) : dash;
    const dttDepStr = (f.proSimDttDepSec != null && isFinite(f.proSimDttDepSec)) ? formatSecondsToHHMMSS(f.proSimDttDepSec) : dash;
    const vttDepStr = (f.proSimVttDepSec != null && isFinite(f.proSimVttDepSec)) ? formatSecondsToHHMMSS(f.proSimVttDepSec) : dash;
    const rotDepStr = (f.proSimDepLineupSec != null && isFinite(f.proSimDepLineupSec)) ? formatSecondsToHHMMSS(f.proSimDepLineupSec) : dash;
    const rotArrCell = flightScheduleProSimTimedCell(rotArrStr, 'green');
    const vttArrCell = flightScheduleProSimTimedCell(vttArrStr, 'vttarr');
    const dttArrCell = flightScheduleProSimTimedCell(dttArrStr, 'dttarr');
    const pushbackCell = flightScheduleProSimTimedCell(pushbackStr, 'pushback');
    const dttDepCell = flightScheduleProSimTimedCell(dttDepStr, 'dttdep');
    const vttDepCell = flightScheduleProSimTimedCell(vttDepStr, 'red');
    const rotDepCell = flightScheduleProSimTimedCell(rotDepStr, 'pink');
    const depRunwayId = f.depRunwayId || (f.token && f.token.depRunwayId);
    ensureFlightSplitTerminalDefaults(f);
    const arrTermId = resolveFlightArrTerminalId(f);
    const depTermId = resolveFlightDepTerminalId(f);
    const arrRwRead = escapeHtml(getRunwayDisplayLabelById(arrRunwayId));
    const arrBuildingRead = escapeHtml(getTerminalDisplayLabelById(arrTermId));
    const depBuildingRead = escapeHtml(getTerminalDisplayLabelById(depTermId));
    const depRwRead = escapeHtml(getRunwayDisplayLabelById(depRunwayId));
    function segTimeCell(seg, key, cls) {
      if (!seg) return '<td class="flight-td-time ' + cls + '" data-empty="1">—</td>';
      const m = Number(seg[key]);
      const txt = isFinite(m) ? formatFlightScheduleDateTime(f, m) : '—';
      return '<td class="flight-td-time ' + cls + '" data-sched-min="' + (isFinite(m) ? m : '') + '">' + escapeHtml(txt) + '</td>';
    }
    function eSeriesCell(minVal, labelIdx) {
      const txt = fmtFlightESchedCell(minVal);
      return '<td class="flight-td-time flight-col-e" data-e-series-index="' + labelIdx + '">' + escapeHtml(txt) + '</td>';
    }
    const sCells = segCells.map(function(seg, idx) {
      return [
        segTimeCell(seg, 'sibtMin', 'flight-col-s' + (idx === 0 ? ' flight-col-s-start flight-td-sibt' : '')),
        segTimeCell(seg, 'sobtMin', 'flight-col-s' + (idx === k - 1 ? ' flight-col-s-last' : ''))
      ].join('');
    }).join('');
    const eCells = segCells.map(function(_seg, idx) {
      const eibtList = flightEMinListForSchedule(f, 'eibtMinList', 'eibtSecList', 'eibtMin');
      const eobtList = flightEMinListForSchedule(f, 'eobtMinList', 'eobtSecList', 'eobtMin');
      return [
        eSeriesCell(eibtList[idx] != null ? eibtList[idx] : null, idx + 1),
        eSeriesCell(eobtList[idx] != null ? eobtList[idx] : null, idx + 1)
      ].join('');
    }).join('');
    const apCells = segCells.map(function(seg) {
      const lab = seg ? flightScheduleStandLabelById(seg.standId) : '—';
      return '<td class="flight-td-readonly" data-empty="' + (seg ? '0' : '1') + '">' + escapeHtml(lab) + '</td>';
    }).join('');
    let laTaxiVal = 9;
    if (f.lookaheadTaxi != null && f.lookaheadTaxi !== '' && isFinite(Number(f.lookaheadTaxi))) {
      laTaxiVal = Math.max(0, Math.min(200, Math.floor(Number(f.lookaheadTaxi))));
    }
    const aircraftTypeLabel = ac ? (ac.name || ac.id || '') : (f.aircraftType || '—');
    const codeIcao = (ac && ac.icao) ? ac.icao : (f.code || '—');
    const intDomVal = (String(f.intDom || '').trim().toLowerCase() === 'dom') ? 'Dom' : 'Int';
    const schedDeadlock = flightScheduleRowHasDeadlock(f);
    const trDeadlockCls = schedDeadlock ? ' flight-sched-row-deadlock' : '';
    return '' +
      '<tr class="flight-data-row obj-item' + trDeadlockCls + '" data-id="' + f.id + '">' +
        '<td class="flight-td-reg">' + escapeHtml(f.reg || '') + '</td>' +
        '<td class="flight-td-reg">' + escapeHtml(f.airlineCode || '') + '</td>' +
        '<td class="flight-td-reg">' + escapeHtml(f.flightNumber || '') + '</td>' +
        '<td>' + escapeHtml(aircraftTypeLabel) + '</td>' +
        '<td>' + escapeHtml(codeIcao) + '</td>' +
        '<td class="flight-td-readonly" title="Edit in Int/Dom above when flight is selected">' + escapeHtml(intDomVal) + '</td>' +
        '<td class="flight-td-readonly">' + arrRwRead + '</td>' +
        '<td class="flight-td-arr-ret' + (arrRetFailed ? ' flight-td-arr-ret-failed' : '') + '">' + (arrRetFailed ? 'Failed' : escapeHtml(sampledRetName)) + '</td>' +
        '<td class="flight-td-readonly">' + arrBuildingRead + '</td>' +
        '<td class="flight-td-readonly">' +         depBuildingRead + '</td>' +
        apCells +
        '<td class="flight-td-readonly flight-td-lookahead-taxi">' + String(laTaxiVal) + '</td>' +
        '<td class="flight-td-readonly">' + depRwRead + '</td>' +
        sCells +
        '<td class="flight-td-time flight-col-e flight-col-e-start">' + escapeHtml(eldtStr) + '</td>' +
        eCells +
        '<td class="flight-td-time flight-col-e">' + escapeHtml(etotStr) + '</td>' +
        '<td class="flight-td-time flight-col-e flight-col-rot">' + rotArrCell + '</td>' +
        '<td class="flight-td-time">' + vttArrCell + '</td>' +
        '<td class="flight-td-time">' + dttArrCell + '</td>' +
        '<td class="flight-td-time">' + pushbackCell + '</td>' +
        '<td class="flight-td-time">' + dttDepCell + '</td>' +
        '<td class="flight-td-time">' + vttDepCell + '</td>' +
        '<td class="flight-td-time flight-col-e flight-col-rot">' + rotDepCell + '</td>' +
        '<td class="flight-td-del"><button type="button" class="obj-item-delete" data-del="' + f.id + '">×</button></td>' +
      '</tr>';
  }

  function _buildFlightListRowsHtml(flightsSorted, retStatsAll, apronK) {
    return flightsSorted.map(function(f) {
      return _buildFlightListRowHtml(f, retStatsAll, apronK);
    });
  }

  const FLIGHT_LIST_PATH_YIELD_CHUNK = 6;
  const FLIGHT_LIST_ASYNC_PATH_MIN = 8;
  function _renderFlightListDomAndSchedule(flightsSorted, schedFull, dirtySet, standSet, listEl, cfgEl, retStatsAll, domOpt) {
    const skipGanttRefresh = domOpt && domOpt.skipGanttRefresh;
    const apronK = flightScheduleColumnK();
    const headerRow = _buildFlightListHeaderHtml(apronK);
    const dirtyIds = [];
    dirtySet.forEach(function(id) { if (id != null && id !== '') dirtyIds.push(id); });
    const deferOnlyDirty = false;
    if (schedFull) {
      if (typeof computeScheduledDisplayTimes === 'function') computeScheduledDisplayTimes(state.flights);
    } else {
      if (!deferOnlyDirty && typeof computeScheduledDisplayTimesIncremental === 'function')
        computeScheduledDisplayTimesIncremental(state.flights, dirtySet, standSet);
    }
    flightsSorted.sort((a, b) => (a.sibtMin != null ? a.sibtMin : (a.timeMin != null ? a.timeMin : 0)) - (b.sibtMin != null ? b.sibtMin : (b.timeMin != null ? b.timeMin : 0)));
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
      _flightListMountVirtual(listEl, flightsSorted, retStatsAll, headerRow, apronK);
    } else {
      _flightListTeardownVirtual(listEl);
      const dataRows = _buildFlightListRowsHtml(flightsForDom, retStatsAll, apronK);
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
    _flightListApplyScheduleSelectionHighlightDom(listEl, false);
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
      flightsSorted.sort((a, b) => (a.sibtMin != null ? a.sibtMin : (a.timeMin != null ? a.timeMin : 0)) - (b.sibtMin != null ? b.sibtMin : (b.timeMin != null ? b.timeMin : 0)));
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
    flightsSorted.sort((a, b) => (a.sibtMin != null ? a.sibtMin : (a.timeMin != null ? a.timeMin : 0)) - (b.sibtMin != null ? b.sibtMin : (b.timeMin != null ? b.timeMin : 0)));
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
      if (typeof syncProSimButtonFromDesignerPageState === 'function') syncProSimButtonFromDesignerPageState();
      return;
    }
    const flights = state.flights.slice();
    const stands = allStandsForFlightAssignment();
    if (!flights.length) {
      state.allocGanttWindowStartMin = null;
      ganttEl.innerHTML = '<div style="font-size:11px;color:#9ca3af;">No flights for Gantt.</div>';
      const labEmpty2 = document.getElementById('allocGanttWindowLabel');
      if (labEmpty2) labEmpty2.textContent = '';
      if (typeof syncProSimButtonFromDesignerPageState === 'function') syncProSimButtonFromDesignerPageState();
      return;
    }
    if (!skipPathPrep) {
      if (typeof computeScheduledDisplayTimes === 'function') computeScheduledDisplayTimes(state.flights);
    }

    let intervals = [];
    const intervalFlightIds = new Set();
    function pushGanttIntervalsFromFlight(f) {
      if (!f) return;
      const fid = f.id != null ? String(f.id) : '';
      const t0 = f.sibtMin != null ? f.sibtMin : (f.timeMin != null ? f.timeMin : 0);
      const t1 = f.sobtMin != null ? f.sobtMin : (t0 + (f.dwellMin != null ? f.dwellMin : 0));
      const sldt = f.sldtMin != null ? f.sldtMin : Math.max(0, t0 - SCHED_SIBT_MINUS_SLDT_MIN);
      const stot = f.stotMin != null ? f.stotMin : (t1 + SCHED_STOT_MINUS_SOBT_MIN);
      const eSer = ganttESeriesMinutesFromTimelineMeta(f);
      if (Array.isArray(f.apronStaySegments) && f.apronStaySegments.length > 1 && typeof buildApronStayGanttIntervalsForFlight === 'function') {
        buildApronStayGanttIntervalsForFlight(f, eSer).forEach(function(it) { intervals.push(it); });
        if (fid) intervalFlightIds.add(fid);
        return;
      }
      const eibt = eSer.eibt;
      const eobt = eSer.eobt;
      const eldt = eSer.eldt;
      const etot = eSer.etot;
      const sldtOrig = sldt;
      const sobtOrig = f.sobtMin != null ? f.sobtMin : t1;
      const stotOrig = stot;
      intervals.push({ f, t0, t1, sldt, stot, eibt, eobt, eldt, etot, sldtOrig, sobtOrig, stotOrig, segmentIdx: 0, segmentCount: 1, segmentStandId: f.standId || null });
      if (fid) intervalFlightIds.add(fid);
    }
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
        const k = flightScheduleColumnK();
        const sibtIdx = flightSchedColIndex('sibt', k);
        const sobtIdx = flightSchedColIndex('sobt', k);
        const eldtIdx = flightSchedColIndex('eldt', k);
        const eibtIdx = flightSchedColIndex('eibt', k);
        const eobtIdx = flightSchedColIndex('eobt', k);
        const etotIdx = flightSchedColIndex('etot', k);
        if (tds.length <= etotIdx) return;
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
        const sibt = getMin(sibtIdx);
        const sobt = getMin(sobtIdx);
        const sldt = Math.max(0, sibt - SCHED_SIBT_MINUS_SLDT_MIN);
        const stot = sobt + SCHED_STOT_MINUS_SOBT_MIN;
        const eSer = ganttESeriesMinutesFromTimelineMeta(f);
        const eldt = eSer.eldt != null ? eSer.eldt : getMin(eldtIdx);
        const eibt = eSer.eibt != null ? eSer.eibt : getMin(eibtIdx);
        const eobt = eSer.eobt != null ? eSer.eobt : getMin(eobtIdx);
        const etot = eSer.etot != null ? eSer.etot : getMin(etotIdx);
        if (Array.isArray(f.apronStaySegments) && f.apronStaySegments.length > 1 && typeof buildApronStayGanttIntervalsForFlight === 'function') {
          buildApronStayGanttIntervalsForFlight(f, eSer).forEach(function(it) { intervals.push(it); });
          intervalFlightIds.add(String(f.id));
        } else {
          const t0 = sibt;
          const t1 = sobt || (t0 + (f.dwellMin != null ? f.dwellMin : 0));
          const sldtOrig = sldt;
          const sobtOrig = sobt || t1;
          const stotOrig = stot;
          intervals.push({ f, t0, t1, sldt, stot, eibt, eobt, eldt, etot, sldtOrig, sobtOrig, stotOrig, segmentIdx: 0, segmentCount: 1, segmentStandId: f.standId || null });
          intervalFlightIds.add(String(f.id));
        }
      });
    }
    if (intervals.length && intervalFlightIds.size < flights.length) {
      flights.forEach(function(f) {
        if (!f || f.id == null || intervalFlightIds.has(String(f.id))) return;
        pushGanttIntervalsFromFlight(f);
      });
    }
    if (!intervals.length) {
      flights.forEach(function(f) { pushGanttIntervalsFromFlight(f); });
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
      if (typeof syncProSimButtonFromDesignerPageState === 'function') syncProSimButtonFromDesignerPageState();
      return;
    }
    const baseMinT = Math.max(0, minS - GANTT_PAD_MIN);
    const baseMaxT0 = maxE + GANTT_PAD_MIN;
    const baseMaxT = (baseMaxT0 <= baseMinT) ? (baseMinT + 60) : baseMaxT0;
    const baseSpan = baseMaxT - baseMinT;
    const dataSpan = Math.max(1e-9, baseSpan);
    const visibleSpan = Math.min(GANTT_VISIBLE_WINDOW_MIN, dataSpan);
    const maxWinStart = Math.max(baseMinT, baseMaxT - visibleSpan);
    let winStart = state.allocGanttWindowStartMin;
    if (winStart == null || !isFinite(winStart)) winStart = baseMinT;
    const vpPin = state._allocGanttHandleDragViewportPin;
    if (vpPin && vpPin.active) {
      let w = vpPin.winStart0;
      if (w == null || !isFinite(w)) w = winStart;
      if (w > maxWinStart) w = maxWinStart;
      if (w + visibleSpan < baseMinT - 1e-6) w = Math.min(maxWinStart, baseMinT);
      winStart = w;
    } else {
      winStart = Math.min(Math.max(winStart, baseMinT), maxWinStart);
    }
    state.allocGanttWindowStartMin = winStart;
    const winEnd = winStart + visibleSpan;
    state._allocGanttClamp = { baseMinT: baseMinT, baseMaxT: baseMaxT, visibleSpan: visibleSpan };
    const displaySpan = visibleSpan;
    const zoomRaw = (state.allocTimeZoom && state.allocTimeZoom > 1) ? state.allocTimeZoom : 1;
    const innerMinWidthPct = Math.max(100, Math.round(zoomRaw * 100));
    const zoomLayout = innerMinWidthPct / 100;
    state._allocGanttPlayheadCtx = { winStart: winStart, winEnd: winEnd, displaySpan: displaySpan, zoom: zoomLayout };

    const tickPositions = buildTimeAxisTicks(winStart, winEnd, winStart, displaySpan, zoomLayout);
