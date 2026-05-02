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

    const tickPositions = buildTimeAxisTicks(winStart, winEnd, winStart, displaySpan, zoomLayout);

    function allocLeftPct(t) {
      return ((t - winStart) / displaySpan) * 100 * zoomLayout;
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
      arr.push(allocTrackSpanHtml(cls, allocLeftPct(clippedStart), ((clippedEnd - clippedStart) / displaySpan) * 100 * zoomLayout, minWidthPct));
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
        const raw = it.segmentStandId != null ? it.segmentStandId : (it.f && it.f.standId);
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
      const duplicateBg = [];
      if (standId != null) {
        const dupIds = duplicateApronStandIdsForStand(standId);
        for (let di = 0; di < dupIds.length; di++) {
          const dupFlights = intervalsByStandKey[String(dupIds[di])] || [];
          for (let ii = 0; ii < dupFlights.length; ii++) {
            const dit = dupFlights[ii];
            if (dit && isFinite(dit.t0) && isFinite(dit.t1) && dit.t1 > dit.t0) {
              pushAllocSpan(duplicateBg, dit.t0, dit.t1, 'alloc-duplicate-bg', 0.5);
            }
          }
        }
      }
      const conflictMap = {};
      for (let i = 0; i < rowFlights.length; i++) {
        for (let j = i + 1; j < rowFlights.length; j++) {
          const a = rowFlights[i];
          const b = rowFlights[j];
          if (a.f && b.f && a.f.id === b.f.id) continue;
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
        const leftPct = ((tStart - winStart) / displaySpan) * 100 * zoomLayout;
        const widthPct = Math.max(2, ((tEnd - tStart) / displaySpan) * 100 * zoomLayout);
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
        const segIdx = it.segmentIdx != null ? Number(it.segmentIdx) : 0;
        const segCount = it.segmentCount != null ? Number(it.segmentCount) : 1;
        const segStandId = it.segmentStandId != null ? it.segmentStandId : standId;
        const segStand = segStandId != null && typeof findStandById === 'function' ? findStandById(segStandId) : null;
        const invalidClass = (segStand && typeof flightCanUseStandForSegment === 'function' && !flightCanUseStandForSegment(f, segStand, segIdx, segCount)) ? ' alloc-invalid' : '';
        const isFirstSeg = segIdx === 0;
        const isLastSeg = segIdx >= segCount - 1;
        const segName = segCount > 1 ? ('AP' + (segIdx + 1)) : '';
        const sibtLabel = formatFlightScheduleDateTime(f, t0);
        const sobtLabel = formatFlightScheduleDateTime(f, t1);
        const handleHoverSibt = escapeAttr((segCount > 1 ? ('SIBT' + (segIdx + 1)) : 'SIBT') + ': ' + sibtLabel);
        const handleHoverSobt = escapeAttr((segCount > 1 ? ('SOBT' + (segIdx + 1)) : 'SOBT') + ': ' + sobtLabel);
        const barTitle =
          (segName ? (segName + '\\n') : '') +
          (segCount > 1 ? ('SIBT' + (segIdx + 1)) : 'SIBT') + ': ' + sibtLabel +
          '\\n' + (segCount > 1 ? ('SOBT' + (segIdx + 1)) : 'SOBT') + ': ' + sobtLabel +
          '\\nReg: ' + (f.reg || '') +
          '\\nAirline: ' + (f.airlineCode || '') + ' ' + (f.flightNumber || '');
        if (showEibtBars && eBars && (it.eBarSegmented || isFirstSeg) && isFinite(eibt) && isFinite(eobt) && eobt > eibt) {
          pushAllocSpan(eBars, eibt, eobt, 'alloc-e-bar', 2);
        }
        if (showEldtBars && e2Bars && isFirstSeg) {
          if (isFinite(eldt) && isFinite(eibt) && eibt >= eldt) pushAllocSpan(e2Bars, eldt, eibt, 'alloc-e2-bar', 0.5);
          if (isFinite(eobt) && isFinite(etot) && etot >= eobt) pushAllocSpan(e2Bars, eobt, etot, 'alloc-e2-bar', 0.5);
        }
        if (showAuxBars && sBars) {
          if (isFirstSeg && isFinite(sldt) && sldt <= t0) pushAllocSpan(sBars, sldt, t0, 'alloc-s-bar', 0.5);
          if (isLastSeg && isFinite(stot) && stot >= t1) pushAllocSpan(sBars, t1, stot, 'alloc-s-bar', 0.5);
        }
        if (showSDots && sDots) {
          if (isFirstSeg) pushAllocDot(sDots, sldt, 'alloc-time-dot-s');
          if (isLastSeg) pushAllocDot(sDots, stot, 'alloc-time-dot-s');
        }
        if (showSdDots && sdDots) {
          if (isFirstSeg) pushAllocDot(sdDots, sldt, 'alloc-time-dot-sd');
          if (isLastSeg) pushAllocDot(sdDots, stot, 'alloc-time-dot-sd');
        }
        if (showEDots && eDots && isFirstSeg) {
          pushAllocDot(eDots, eldt, 'alloc-time-dot-e');
          pushAllocDot(eDots, etot, 'alloc-time-dot-e');
          pushAllocTriangle(eTrisDown, eldt, 'alloc-e-tri alloc-e-tri-down');
          pushAllocTriangle(eTrisUp, etot, 'alloc-e-tri alloc-e-tri-up');
        }
        if (showSPoints) {
          if (isFirstSeg) pushAllocTriangle(sTrisDown, sldt, 'alloc-s-tri alloc-s-tri-down');
          if (isLastSeg) pushAllocTriangle(sTrisUp, stot, 'alloc-s-tri alloc-s-tri-up');
        }
        const blocked = typeof flightBlockedLikeNoWay === 'function' && flightBlockedLikeNoWay(f);
        const handleParts = [];
        if (!blocked) {
          const segsForHandles = Array.isArray(f.apronStaySegments) ? f.apronStaySegments : [];
          const leftUnlocked = isFirstSeg || !(_sameApronStayStand(segsForHandles[segIdx - 1], segsForHandles[segIdx]));
          const rightUnlocked = isLastSeg || !(_sameApronStayStand(segsForHandles[segIdx], segsForHandles[segIdx + 1]));
          if (leftUnlocked) handleParts.push('<button type="button" class="alloc-flight-handle alloc-flight-handle--sibt" data-handle-role="sibt" data-segment-idx="' + segIdx + '" tabindex="-1" aria-label="Adjust SIBT (5 min)" title="' + handleHoverSibt + '"></button>');
          else handleParts.push('<span class="alloc-flight-junction alloc-flight-junction--left alloc-flight-junction--locked" title="Move to a different stand to enable timing edits"></span>');
          if (rightUnlocked) handleParts.push('<button type="button" class="alloc-flight-handle alloc-flight-handle--sobt" data-handle-role="sobt" data-segment-idx="' + segIdx + '" tabindex="-1" aria-label="Adjust SOBT (5 min)" title="' + handleHoverSobt + '"></button>');
          else handleParts.push('<span class="alloc-flight-junction alloc-flight-junction--right alloc-flight-junction--locked" title="Move to a different stand to enable timing edits"></span>');
        }
        const splitHtml = (!blocked && (t1 - t0) >= (APRON_STAY_SPLIT_MIN_PART_MIN * 2))
          ? '<button type="button" class="alloc-flight-split-btn" data-segment-idx="' + segIdx + '" title="Split apron stay">Split</button>'
          : '';
        const segBadgeHtml = segName ? '<span class="alloc-flight-ap-badge">' + escapeHtml(segName) + '</span>' : '';
        return '' +
          '<div class="alloc-flight' + conflictClass + invalidClass + selectedClass + sbarDimClass + '" draggable="true" data-flight-id="' + f.id + '" data-segment-idx="' + segIdx + '" ' +
            'style="left:' + leftPct + '%;width:' + widthPct + '%;min-width:4px;"' +
            ' title="' + barTitle + '">' +
            handleParts.join('') +
            '<div class="alloc-flight-reg">' + regSafe + '</div>' +
            metaHtml +
            segBadgeHtml +
            splitHtml +
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
            duplicateBg.join('') +
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
    const labelRows = [];
    const trackRows = [];
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
    function syncAllocLabelToScrollCol() {
      if (newScrollCol && newLabelCol) {
        newLabelCol.scrollTop = newScrollCol.scrollTop;
      }
    }
    if (newScrollCol) {
      newScrollCol.scrollLeft = prevScrollLeft;
      newScrollCol.scrollTop = prevScrollTop;
    }
    syncAllocLabelToScrollCol();
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
    syncAllocLabelToScrollCol();
    requestAnimationFrame(syncAllocLabelToScrollCol);
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
    if (typeof syncProSimButtonFromDesignerPageState === 'function') syncProSimButtonFromDesignerPageState();
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

  function _ganttClientXToMinutes(clientX, ganttEl) {
    const scrollCol = ganttEl.querySelector('.alloc-gantt-scroll-col');
    const inner = scrollCol && scrollCol.querySelector('.alloc-gantt-inner');
    if (!scrollCol || !inner) return null;
    const rect = inner.getBoundingClientRect();
    const x = clientX - rect.left + scrollCol.scrollLeft;
    const w = inner.scrollWidth;
    if (!(w > 0)) return null;
    const frac = Math.max(0, Math.min(1, x / w));
    const c = state._allocGanttClamp;
    const winStart = state.allocGanttWindowStartMin;
    if (!c || winStart == null || !isFinite(winStart)) return null;
    return winStart + frac * c.visibleSpan;
  }

  function _ganttEnsureSibtSobtTooltipEl() {
    let el = document.getElementById('allocGanttSibtSobtTooltip');
    if (!el) {
      el = document.createElement('div');
      el.id = 'allocGanttSibtSobtTooltip';
      el.className = 'alloc-gantt-sibt-sobt-tooltip';
      el.setAttribute('hidden', '');
      document.body.appendChild(el);
    }
    return el;
  }

  var _allocGanttPreviewTimer = null;
