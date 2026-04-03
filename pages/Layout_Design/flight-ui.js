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
  function sampleArrRetRotForFlightIfNeeded(f, retStatsAll, configByType, forceResample) {
    if (!f) return;
    const rev = state.vttArrCacheRev | 0;
    if (!forceResample && f.deferPathCompute) {
      f.__schedRetRotRev = rev;
      return;
    }
    if (!forceResample && f.__schedRetRotRev === rev && isValidSampledArrRetForFlight(f, retStatsAll)) return;
    if (!forceResample && (f.__schedRetRotRev === undefined || f.__schedRetRotRev === null) &&
        f.sampledArrRet != null && f.arrRetFailed === false && f.arrRotSec != null && isFinite(f.arrRotSec) &&
        isValidSampledArrRetForFlight(f, retStatsAll)) {
      f.__schedRetRotRev = rev;
      return;
    }
    if (f.sampledArrRet != null && !isValidSampledArrRetForFlight(f, retStatsAll)) {
      f.sampledArrRet = null;
      f.arrRetFailed = false;
      f.arrRotSec = null;
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
      const tToRetEntrance = rtRunway.tSec;
      const minExitVel = (typeof chosen.minExitVelocity === 'number' && isFinite(chosen.minExitVelocity) && chosen.minExitVelocity > 0)
        ? Math.min(chosen.minExitVelocity, chosen.maxExitVelocity || chosen.minExitVelocity)
        : 15;
      let tExit = 0;
      if (vAtChosen > minExitVel) {
        tExit = (vAtChosen - minExitVel) / aDecRot;
      }
      f.arrRotSec = tToRetEntrance + tExit;
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
      f.arrRotSec = null;
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
        '<th class="flight-col-s flight-col-s-start">SLDT</th>' +
        '<th class="flight-td-sibt flight-col-s">SIBT</th>' +
        '<th class="flight-col-s">SOBT</th>' +
        '<th class="flight-col-s flight-col-s-last">STOT</th>' +
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
    if (f.deferPathCompute) {
      const tArrMin = f.timeMin != null ? f.timeMin : 0;
      const dwell = f.dwellMin != null ? f.dwellMin : 0;
      const tDepMin = tArrMin + dwell;
      const tArr = formatMinutesToHHMMSS(tArrMin);
      const tDep = formatMinutesToHHMMSS(tDepMin);
      const dash = '—';
      const depRunwayId = f.depRunwayId || (f.token && f.token.depRunwayId);
      const termId = f.terminalId || (f.token && f.token.terminalId);
      const arrRwRead = escapeHtml(getRunwayDisplayLabelById(arrRunwayId));
      const buildingRead = escapeHtml(getTerminalDisplayLabelById(termId));
      const depRwRead = escapeHtml(getRunwayDisplayLabelById(depRunwayId));
      const aircraftTypeLabel = ac ? (ac.name || ac.id || '') : (f.aircraftType || '—');
      const codeIcao = (ac && ac.icao) ? ac.icao : (f.code || '—');
      const pathPendingClass = ' flight-row-path-pending';
      const pathPendingTitle = ' title="' + escapeAttr('경로 미계산 — Update로 반영') + '"';
      return '' +
        '<tr class="flight-data-row obj-item' + pathPendingClass + '"' + pathPendingTitle + ' data-id="' + f.id + '">' +
          '<td class="flight-td-reg">' + escapeHtml(f.reg || '') + '</td>' +
          '<td class="flight-td-reg">' + escapeHtml(f.airlineCode || '') + '</td>' +
          '<td class="flight-td-reg">' + escapeHtml(f.flightNumber || '') + '</td>' +
          '<td class="flight-td-readonly">' + arrRwRead + '</td>' +
          '<td class="flight-td-arr-ret' + (arrRetFailed ? ' flight-td-arr-ret-failed' : '') + '">' + (arrRetFailed ? 'Failed' : escapeHtml(sampledRetName)) + '</td>' +
          '<td class="flight-td-readonly">' + buildingRead + '</td>' +
          '<td class="flight-td-reg">' + (function() { var st = findStandById(f.standId); return escapeHtml(st ? ((st.name && st.name.trim()) || st.id || '—') : '—'); })() + '</td>' +
          '<td class="flight-td-readonly">' + depRwRead + '</td>' +
          '<td class="flight-td-time flight-col-s flight-col-s-start">' + dash + '</td>' +
          '<td class="flight-td-time flight-td-sibt flight-col-s">' + tArr + '</td>' +
          '<td class="flight-td-time flight-col-s">' + tDep + '</td>' +
          '<td class="flight-td-time flight-col-s flight-col-s-last">' + dash + '</td>' +
          '<td class="flight-td-time flight-col-sd flight-col-sd-start">' + dash + '</td>' +
          '<td class="flight-td-time flight-col-sd">' + dash + '</td>' +
          '<td class="flight-td-time flight-col-sd">' + dash + '</td>' +
          '<td class="flight-td-time flight-col-sd flight-col-sd-last">' + dash + '</td>' +
          '<td class="flight-td-time flight-col-e flight-col-e-start">' + dash + '</td>' +
          '<td class="flight-td-time flight-col-e">' + dash + '</td>' +
          '<td class="flight-td-time flight-col-e">' + dash + '</td>' +
          '<td class="flight-td-time flight-col-e">' + dash + '</td>' +
          '<td class="flight-td-time flight-col-e flight-col-rot">' + dash + '</td>' +
          '<td class="flight-td-time">' + dash + '</td>' +
          '<td class="flight-td-time">' + dash + '</td>' +
          '<td class="flight-td-time">' + dash + '</td>' +
          '<td class="flight-td-time">' + dash + '</td>' +
          '<td class="flight-td-time">' + dash + '</td>' +
          '<td>' + escapeHtml(aircraftTypeLabel) + '</td>' +
          '<td>' + escapeHtml(codeIcao) + '</td>' +
          '<td class="flight-td-del"><button type="button" class="obj-item-delete" data-del="' + f.id + '">×</button></td>' +
        '</tr>';
    }
    const tArrMin = f.timeMin != null ? f.timeMin : 0;
    const dwell = f.dwellMin != null ? f.dwellMin : 0;
    const tDepMin = tArrMin + dwell;
    const vttArrMin = getBaseVttArrMinutes(f);
    const rotArrMin = getArrRotMinutes(f);
    const depBlockOutMin = (typeof getDepBlockOutMin === 'function') ? getDepBlockOutMin(f) : 0;
    const rollBundleSecFallback = DEP_LINEUP_HOLD_SEC + takeoffRollSecForRunwayTailLenM(0, DEP_TAKEOFF_ACCEL_SMALL_MS2);
    const vttDepMinLineup = (typeof getBaseVttDepMinutesToLineup === 'function')
      ? getBaseVttDepMinutesToLineup(f)
      : Math.max(0, depBlockOutMin - ((typeof computeDepRollAndLineupOnlySec === 'function') ? computeDepRollAndLineupOnlySec(f) : rollBundleSecFallback) / 60);
    const vttDepMinSlot = (typeof getBaseVttDepMinutesToHoldingSlot === 'function') ? getBaseVttDepMinutesToHoldingSlot(f) : vttDepMinLineup;
    const depRotSecVal = (typeof computeDepRotSecondsForFlight === 'function') ? computeDepRotSecondsForFlight(f) : Math.max(0, Number(SCHED_DEP_ROT_MIN) || 2) * 60;
    const rotDepMin = depRotSecVal / 60;
    const sldtCalc = (f.sldtMin_d != null ? f.sldtMin_d : Math.max(0, tArrMin - vttArrMin - rotArrMin));
    const sldtOrig = f.sldtMin_orig != null ? f.sldtMin_orig : sldtCalc;
    const sobtOrig = (f.sobtMin_orig != null) ? f.sobtMin_orig : tDepMin;
    const stotOrig = (f.stotMin_orig != null) ? f.stotMin_orig : (tDepMin + rotDepMin + vttDepMinSlot);
    const sldtStr = formatMinutesToHHMMSS(f.sldtMin_orig != null ? f.sldtMin_orig : sldtCalc);
    const stotStr = formatMinutesToHHMMSS(stotOrig);
    const sldtStr_d = formatMinutesToHHMMSS(f.sldtMin_d != null ? f.sldtMin_d : sldtOrig);
    const sibtStr_d = formatMinutesToHHMMSS(f.sibtMin_d != null ? f.sibtMin_d : tArrMin);
    const sobtStr_d = formatMinutesToHHMMSS(f.sobtMin_d != null ? f.sobtMin_d : tDepMin);
    const stotStr_d = formatMinutesToHHMMSS(f.stotMin_d != null ? f.stotMin_d : stotOrig);
    const eldtMin = f.eldtMin != null ? f.eldtMin : (f.sldtMin_d != null ? f.sldtMin_d : sldtOrig);
    const etotCandMin = f.etotMin != null ? f.etotMin : (f.stotMin_d != null ? f.stotMin_d : stotOrig);
    f.eldtMin = eldtMin;
    const tArr = formatMinutesToHHMMSS(tArrMin);
    const tDep = formatMinutesToHHMMSS(tDepMin);
    const vttADelayMin = f.vttADelayMin != null ? f.vttADelayMin : 0;
    const eibtMin = eldtMin + rotArrMin + vttArrMin + vttADelayMin;
    f.eibtMin = eibtMin;
    applyForwardEobtEtotAndDepTaxiDelay(f, eibtMin, etotCandMin);
    const eobtMin = f.eobtMin != null ? f.eobtMin : (f.etotMin != null ? f.etotMin - depBlockOutMin : 0);
    const etotMin = f.etotMin != null ? f.etotMin : (eobtMin + depBlockOutMin);
    if (f.sobtMin_orig == null) {
      f.sldtMin_orig = sldtOrig;
      f.sibtMin_orig = tArrMin;
      f.sobtMin_orig = sobtOrig;
      f.stotMin_orig = stotOrig;
      f.eldtMin_orig = eldtMin;
      f.eibtMin_orig = eibtMin;
      f.eobtMin_orig = eobtMin;
      f.etotMin_orig = etotMin;
    }
    const eldtStr = formatMinutesToHHMMSS(eldtMin);
    const etotStr = formatMinutesToHHMMSS(etotMin);
    const eibtStr = formatMinutesToHHMMSS(eibtMin);
    const eobtStr = formatMinutesToHHMMSS(eobtMin);
    const vttArrStr = formatMinutesToHHMMSS(vttArrMin);
    const vttADelayStr = formatMinutesToHHMMSS(vttADelayMin);
    const vttDepStr = formatMinutesToHHMMSS(vttDepMinSlot);
    const depRotStr = formatTotalSecondsToHHMMSS(depRotSecVal);
    const depTaxiDelayStr = formatSignedMinutesToHHMMSS(f.depTaxiDelayMin != null ? f.depTaxiDelayMin : 0);
    const depRunwayId = f.depRunwayId || (f.token && f.token.depRunwayId);
    const termId = f.terminalId || (f.token && f.token.terminalId);
    const arrRwRead = escapeHtml(getRunwayDisplayLabelById(arrRunwayId));
    const buildingRead = escapeHtml(getTerminalDisplayLabelById(termId));
    const depRwRead = escapeHtml(getRunwayDisplayLabelById(depRunwayId));
    const aircraftTypeLabel = ac ? (ac.name || ac.id || '') : (f.aircraftType || '—');
    const codeIcao = (ac && ac.icao) ? ac.icao : (f.code || '—');
    const pathPendingClass = f.deferPathCompute ? ' flight-row-path-pending' : '';
    const pathPendingTitle = f.deferPathCompute ? ' title="' + escapeAttr('경로 미계산 — Update로 반영') + '"' : '';
    return '' +
      '<tr class="flight-data-row obj-item' + pathPendingClass + '"' + pathPendingTitle + ' data-id="' + f.id + '">' +
        '<td class="flight-td-reg">' + escapeHtml(f.reg || '') + '</td>' +
        '<td class="flight-td-reg">' + escapeHtml(f.airlineCode || '') + '</td>' +
        '<td class="flight-td-reg">' + escapeHtml(f.flightNumber || '') + '</td>' +
        '<td class="flight-td-readonly">' + arrRwRead + '</td>' +
        '<td class="flight-td-arr-ret' + (arrRetFailed ? ' flight-td-arr-ret-failed' : '') + '">' + (arrRetFailed ? 'Failed' : escapeHtml(sampledRetName)) + '</td>' +
        '<td class="flight-td-readonly">' + buildingRead + '</td>' +
        '<td class="flight-td-reg">' + (function() { var st = findStandById(f.standId); return escapeHtml(st ? ((st.name && st.name.trim()) || st.id || '—') : '—'); })() + '</td>' +
        '<td class="flight-td-readonly">' + depRwRead + '</td>' +
        '<td class="flight-td-time flight-col-s flight-col-s-start flight-sched-s-edit" contenteditable="true" spellcheck="false" data-s-field="sldt" data-flight-id="' + escapeAttr(String(f.id)) + '">' + escapeHtml(sldtStr) + '</td>' +
        '<td class="flight-td-time flight-td-sibt flight-col-s flight-sched-s-edit" contenteditable="true" spellcheck="false" data-s-field="sibt" data-flight-id="' + escapeAttr(String(f.id)) + '">' + escapeHtml(tArr) + '</td>' +
        '<td class="flight-td-time flight-col-s flight-sched-s-edit" contenteditable="true" spellcheck="false" data-s-field="sobt" data-flight-id="' + escapeAttr(String(f.id)) + '">' + escapeHtml(tDep) + '</td>' +
        '<td class="flight-td-time flight-col-s flight-col-s-last flight-sched-s-edit" contenteditable="true" spellcheck="false" data-s-field="stot" data-flight-id="' + escapeAttr(String(f.id)) + '">' + escapeHtml(stotStr) + '</td>' +
        '<td class="flight-td-time flight-col-sd flight-col-sd-start">' + sldtStr_d + '</td>' +
        '<td class="flight-td-time flight-col-sd">' + sibtStr_d + '</td>' +
        '<td class="flight-td-time flight-col-sd">' + sobtStr_d + '</td>' +
        '<td class="flight-td-time flight-col-sd flight-col-sd-last">' + stotStr_d + '</td>' +
        '<td class="flight-td-time flight-col-e flight-col-e-start">' + eldtStr + '</td>' +
        '<td class="flight-td-time flight-col-e">' + eibtStr + '</td>' +
        '<td class="flight-td-time flight-col-e">' + eobtStr + '</td>' +
        '<td class="flight-td-time flight-col-e">' + etotStr + '</td>' +
        '<td class="flight-td-time flight-col-e flight-col-rot">' + (f.arrRotSec != null && isFinite(f.arrRotSec) ? (Math.round(f.arrRotSec) + ' s') : '—') + '</td>' +
        '<td class="flight-td-time">' + vttArrStr + '</td>' +
        '<td class="flight-td-time">' + vttADelayStr + '</td>' +
        '<td class="flight-td-time">' + depRotStr + '</td>' +
        '<td class="flight-td-time">' + vttDepStr + '</td>' +
        '<td class="flight-td-time">' + depTaxiDelayStr + '</td>' +
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
    const deferOnlyDirty = dirtyIds.length > 0 && dirtyIds.every(function(fid) {
      const ff = flightsSorted.find(function(x) { return x.id === fid; });
      return ff && ff.deferPathCompute;
    });
    if (schedFull) {
      if (typeof computeScheduledDisplayTimes === 'function') computeScheduledDisplayTimes(state.flights);
      if (typeof computeSeparationAdjustedTimes === 'function') computeSeparationAdjustedTimes();
      pinEarliestEldtToSldtPerRunway(flightsSorted);
    } else {
      if (!deferOnlyDirty && typeof computeScheduledDisplayTimesIncremental === 'function')
        computeScheduledDisplayTimesIncremental(state.flights, dirtySet, standSet);
      if (!deferOnlyDirty && typeof computeSeparationAdjustedTimes === 'function') computeSeparationAdjustedTimes();
      if (!deferOnlyDirty && typeof pinEarliestEldtToSldtPerRunway === 'function') pinEarliestEldtToSldtPerRunway(flightsSorted);
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
      const dirtyForRet = dirtyFlights.filter(function(f) { return f && !f.deferPathCompute; });
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
      _renderFlightListAfterPathEnsure(flightsSorted, schedFull, forceResampleRet, dirtySet, standSet, listEl, cfgEl);
      if (cb) cb();
    }
    const useBatchedPathEnsure = schedFull && cb && flightsSorted.length >= FLIGHT_LIST_ASYNC_PATH_MIN;
    if (useBatchedPathEnsure) {
      let idx = 0;
      function pathChunk() {
        const end = Math.min(idx + FLIGHT_LIST_PATH_YIELD_CHUNK, flightsSorted.length);
        for (; idx < end; idx++) ensureFlightPaths(flightsSorted[idx]);
        if (idx < flightsSorted.length) setTimeout(pathChunk, 0);
        else runTail();
      }
      setTimeout(pathChunk, 0);
      return;
    }
    if (schedFull) {
      flightsSorted.forEach(function(f) { ensureFlightPaths(f); });
    } else {
      dirtySet.forEach(function(fid) {
        const ff = flightsSorted.find(function(x) { return x.id === fid; });
        if (ff) ensureFlightPaths(ff);
      });
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
    const retStats = typeof computeRunwayExitDistances === 'function' ? computeRunwayExitDistances() : [];
    if (retStats && retStats.length) {
      rows.push(
        '<tr>' +
          '<td class="sticky-col" style="padding-top:10px;">Runway exits (distance from threshold)</td>' +
          '<td></td>' +
          '<td></td>' +
          unique.map(() => '<td></td>').join('') +
        '</tr>'
      );
      retStats.forEach((r, idx) => {
        const rwLabel = r.runway && (r.runway.name || ('Runway ' + (idx + 1)));
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
                (rwLabel ? ('<span style="font-size:9px;color:#9ca3af;">' + escapeHtml(rwLabel) + '</span>') : '') +
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
      const trCls = f.deferPathCompute ? ' class="flight-config-sampled-row-pending"' : '';
      return '' +
        '<tr' + trCls + '>' +
          '<td>' + escapeHtml(f.reg || '—') + '</td>' +
          '<td>' + escapeHtml(f.airlineCode || '—') + '</td>' +
          '<td>' + escapeHtml(f.flightNumber || '—') + '</td>' +
          '<td>' + escapeHtml(String(typeLabel)) + '</td>' +
          '<td style="text-align:right;font-variant-numeric:tabular-nums;">' + _fmtFlightPhysVal(f.arrVTdMs) + '</td>' +
          '<td style="text-align:right;font-variant-numeric:tabular-nums;">' + _fmtFlightPhysVal(f.arrDecelMs2) + '</td>' +
        '</tr>';
    }).join('');
    const perFlightBlock = '' +
      '<div class="flight-config-sampled-caption">' +
        '<span class="flight-config-sampled-caption-ko">항공기별 적용값 · 샘플링된 접지속도(VTD)와 활주로 감속도</span>' +
        '<span class="flight-config-sampled-caption-en">Per flight: sampled VTD &amp; deceleration (used after Update / path compute)</span>' +
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
          '</tr></thead>' +
          '<tbody>' + perFlightBody + '</tbody>' +
        '</table>' +
      '</div>';
    cfgEl.innerHTML = cfgHeader + rows.join('') + '</tbody></table>' +
      '<div style="font-size:10px;color:#6b7280;margin-top:4px;">' +
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

  function ensureFlightSchedSColumnEditWired(listEl) {
    if (!listEl || listEl._flightSchedSWired) return;
    listEl._flightSchedSWired = true;
    listEl.addEventListener('focusin', function(ev) {
      var el = ev.target;
      if (!el.classList || !el.classList.contains('flight-sched-s-edit')) return;
      el.setAttribute('data-s-prev', (el.textContent || '').trim());
    });
    listEl.addEventListener('keydown', function(ev) {
      var el = ev.target;
      if (!el.classList || !el.classList.contains('flight-sched-s-edit')) return;
      if (ev.key === 'Enter') {
        ev.preventDefault();
        el.blur();
      } else if (ev.key === 'Escape') {
        ev.preventDefault();
        var p = el.getAttribute('data-s-prev');
        if (p != null) el.textContent = p;
        el.blur();
      }
    });
    listEl.addEventListener('mousedown', function(ev) {
      if (ev.target.classList && ev.target.classList.contains('flight-sched-s-edit')) ev.stopPropagation();
    }, true);
    listEl.addEventListener('dblclick', function(ev) {
      if (ev.target.classList && ev.target.classList.contains('flight-sched-s-edit')) {
        ev.preventDefault();
        ev.stopPropagation();
      }
    }, true);
    listEl.addEventListener('focusout', function(ev) {
      var el = ev.target;
      if (!el.classList || !el.classList.contains('flight-sched-s-edit')) return;
      var fid = el.getAttribute('data-flight-id');
