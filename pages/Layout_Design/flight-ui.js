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
    const depRotSecVal = (typeof computeDepRotSecondsForFlight === 'function') ? computeDepRotSecondsForFlight(f) : Math.max(0, Number(SCHED_DEP_ROT_MIN) || 2) * 60;
    const depRotStr = formatTotalSecondsToHHMMSS(depRotSecVal);
    const depTaxiDelayStr = formatSignedMinutesToHHMMSS(f.depTaxiDelayMin != null ? f.depTaxiDelayMin : 0);
    const depRunwayId = f.depRunwayId || (f.token && f.token.depRunwayId);
    const termId = f.terminalId || (f.token && f.token.terminalId);
    const arrRwRead = escapeHtml(getRunwayDisplayLabelById(arrRunwayId));
    const buildingRead = escapeHtml(getTerminalDisplayLabelById(termId));
    const depRwRead = escapeHtml(getRunwayDisplayLabelById(depRunwayId));
    const noWayBadge = flightBlockedLikeNoWay(f)
      ? ' <span class="flight-no-way-badge" style="color:#dc2626;font-weight:600;font-size:10px;cursor:help;" title="' + escapeAttr(buildNoWayTooltip(f)) + '">⚠ No Way</span>'
      : '';
    const aircraftTypeLabel = ac ? (ac.name || ac.id || '') : (f.aircraftType || '—');
    const codeIcao = (ac && ac.icao) ? ac.icao : (f.code || '—');
    const arrRetFailedBadge = (sampledRetName === 'Failed' && !flightBlockedLikeNoWay(f)) ? ' <span style="color:#dc2626;font-weight:600;font-size:10px;">⚠ Failed</span>' : '';
    const pathPendingClass = f.deferPathCompute ? ' flight-row-path-pending' : '';
    const pathPendingTitle = f.deferPathCompute ? ' title="' + escapeAttr('경로 미계산 — Update로 반영') + '"' : '';
    return '' +
      '<tr class="flight-data-row obj-item' + pathPendingClass + '"' + pathPendingTitle + ' data-id="' + f.id + '">' +
        '<td class="flight-td-reg">' + escapeHtml(f.reg || '') + noWayBadge + arrRetFailedBadge + '</td>' +
        '<td class="flight-td-reg">' + escapeHtml(f.airlineCode || '') + '</td>' +
        '<td class="flight-td-reg">' + escapeHtml(f.flightNumber || '') + '</td>' +
        '<td class="flight-td-time flight-col-s flight-col-s-start">' + sldtStr + '</td>' +
        '<td class="flight-td-time flight-td-sibt flight-col-s">' + tArr + '</td>' +
        '<td class="flight-td-time flight-col-s">' + tDep + '</td>' +
        '<td class="flight-td-time flight-col-s flight-col-s-last">' + stotStr + '</td>' +
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
        '<td class="flight-td-readonly">' + arrRwRead + '</td>' +
        '<td>' + escapeHtml(sampledRetName) + '</td>' +
        '<td class="flight-td-readonly">' + buildingRead + '</td>' +
        '<td class="flight-td-reg">' + (function() { var st = findStandById(f.standId); return escapeHtml(st ? ((st.name && st.name.trim()) || st.id || '—') : '—'); })() + '</td>' +
        '<td class="flight-td-readonly">' + depRwRead + '</td>' +
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
    if (schedFull || !deferOnlyDirty) {
      _renderFlightConfigTable(cfgEl, flightsSorted);
    }
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
            (f.aircraftType || '') === typeKey
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
          (f.aircraftType || '') === typeKey
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
    cfgEl.innerHTML = cfgHeader + rows.join('') + '</tbody></table>' +
      '<div style="font-size:10px;color:#6b7280;margin-top:4px;">' +
        'Note: sampling is clipped to stay within ±15% of each mean value.' +
      '</div>';
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
    const stands = (state.pbbStands || []).concat(state.remoteStands || []);
    if (!flights.length) {
      state.allocGanttWindowStartMin = null;
      ganttEl.innerHTML = '<div style="font-size:11px;color:#9ca3af;">No flights for Gantt.</div>';
      const labEmpty2 = document.getElementById('allocGanttWindowLabel');
      if (labEmpty2) labEmpty2.textContent = '';
      return;
    }
    if (!skipPathPrep) {
      flights.forEach(function(f) { ensureFlightPaths(f); });
      if (typeof ensureArrRetRotSampled === 'function') ensureArrRetRotSampled(flights, false);
      if (typeof computeScheduledDisplayTimes === 'function') computeScheduledDisplayTimes(state.flights);
      if (typeof computeSeparationAdjustedTimes === 'function') computeSeparationAdjustedTimes();
    }

    let intervals = [];
    const schedTable = document.querySelector('.flight-schedule-table');
    const domScheduleOk = schedTable && schedTable.getAttribute('data-virtual-table') !== '1';
    if (domScheduleOk) {
      const rows = Array.from(schedTable.querySelectorAll('tbody tr.flight-data-row'));
      rows.forEach(row => {
        const id = row.getAttribute('data-id');
        if (!id) return;
        const f = flights.find(ff => ff.id === id);
        if (!f) return;
        const tds = Array.from(row.querySelectorAll('td'));
        if (tds.length < 15) return;
        const getMin = (idx) => {
          const txt = (tds[idx] && tds[idx].textContent || '').trim();
          if (!txt) return 0;
          try {
            return parseTimeToMinutes(txt);
          } catch (e) {
            return 0;
          }
        };
        const sldt_d = getMin(7);
        const sibt_d = getMin(8);
        const sobt_d = getMin(9);
        const stot_d = getMin(10);
        const eldt   = getMin(11);
        const eibt   = getMin(12);
        const eobt   = getMin(13);
        const etot   = getMin(14);
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
        const eibt = f.eibtMin != null ? f.eibtMin : t0;
        const eobt = f.eobtMin != null ? f.eobtMin : t1;
        const eldt = f.eldtMin != null ? f.eldtMin : sldt;
        const etot = f.etotMin != null ? f.etotMin : stot;
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
      const etot0 = (it.f && it.f.etotMin != null) ? it.f.etotMin : it.stot;
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
    const baseMaxT = Math.min(
      (baseMaxT0 <= baseMinT) ? (baseMinT + 60) : baseMaxT0,
      baseMinT + 1440
    );
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
      const rowFlights = intervals.filter(it => {
        const f = it.f;
        const sid = (f.standId || null);
        return (standId == null) ? !sid : sid === standId;
      });
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
        const dwellVal = (t1 != null && t0 != null) ? Math.max(0, t1 - t0) : (f.dwellMin != null ? f.dwellMin : 0);
        const dwellLabel = dwellVal ? (Math.round(dwellVal * 10) / 10 + 'm') : '';
        let meta = '';
        if (codeSafe && dwellLabel) meta = codeSafe + ' · ' + dwellLabel;
        else if (codeSafe) meta = codeSafe;
        else meta = dwellLabel;
        const conflictClass = (conflictMap[f.id] || flightBlockedLikeNoWay(f)) ? ' conflict' : '';
        const selectedClass = (state.selectedObject && state.selectedObject.type === 'flight' && state.selectedObject.id === f.id) ? ' alloc-flight-selected' : '';
        const sbarDimClass = dimSBars ? ' alloc-flight-sbar-dim' : '';
        const noWayLabel = (flightBlockedLikeNoWay(f))
          ? ' <span class="flight-no-way-badge" style="color:#fca5a5;font-size:9px;font-weight:700;cursor:help;" title="' + escapeAttr(buildNoWayTooltip(f)) + '">No way</span>'
          : '';
        const sibtLabel = formatMinutesToHHMM(t0);
        const sobtLabel = formatMinutesToHHMM(t1);
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
            '<div class="alloc-flight-reg">' + regSafe + noWayLabel + '</div>' +
            '<div class="alloc-flight-meta">' + meta + '</div>' +
            ovlpBadgeHtml +
          '</div>';
      }).join('');
      const sidAttr = standId ? String(standId) : '';
      const bgSlots = (tickPositions.length > 1)
        ? tickPositions.slice(0, -1).map((tp, idx) => {
            const next = tickPositions[idx + 1];
            const midLeft = (tp.leftPct + next.leftPct) / 2;
            return (
              '<div class="alloc-apron-bg-slot" style="left:' + midLeft + '%;transform:translateX(-50%);">' +
                escapeHtml(label) +
              '</div>'
