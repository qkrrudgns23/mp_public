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
      return '' +
        '<tr>' +
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
    const stands = (state.pbbStands || []).concat(state.remoteStands || []);
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
      rows.forEach(row => {
        const id = row.getAttribute('data-id');
        if (!id) return;
        const f = flights.find(ff => ff.id === id);
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
        const sldt_d = getMin(FLIGHT_SCHED_TD_SLD);
        const sibt_d = getMin(FLIGHT_SCHED_TD_SIBTD);
        const sobt_d = getMin(FLIGHT_SCHED_TD_SOBTD);
        const stot_d = getMin(FLIGHT_SCHED_TD_STOTD);
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
            '<div class="alloc-flight-meta">' + meta + '</div>' +
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
    const grouped = {};
    const order = [];
    const sortedStands = stands.slice().sort((a, b) => {
      const ta = getTerminalForStand(a);
      const tb = getTerminalForStand(b);
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
      const term = getTerminalForStand(s);
      const key = term ? term.id : '__no_terminal__';
      if (!grouped[key]) {
        grouped[key] = { term, stands: [] };
        order.push(key);
      }
      grouped[key].stands.push(s);
    });
    const remoteIdSet = new Set((state.remoteStands || []).map(r => r.id));
    const allRemoteStands = [];
    order.forEach(key => {
      const group = grouped[key];
      if (!group) return;
      const term = group.term;
      const headerLabel = term
        ? (termLabelById[term.id] || term.name || 'Building')
        : 'No Building';
      labelRows.push(
        '<div class="alloc-terminal-header" data-collapsed="0">' +
          '<span class="alloc-section-toggle-icon">▼</span>' +
          escapeHtml(headerLabel) +
        '</div>'
      );
      trackRows.push('<div class="alloc-row" data-stand-id="">' +
        '<div class="alloc-row-track" data-stand-id="" style="background:transparent;border:none;height:24px;"></div>' +
      '</div>');
      const contactStands = [];
      const remoteStandsInTerm = [];
      group.stands.forEach(s => {
        if (remoteIdSet.has(s.id)) remoteStandsInTerm.push(s);


        else contactStands.push(s);
      });
      contactStands.forEach(s => {
        const label = (s.name || '') + ' (' + (s.category || '') + ')';
        const row = buildRowHtml(label, s.id);
        labelRows.push(row.labelHtml);
        trackRows.push(row.trackHtml);
      });
      if (remoteStandsInTerm.length) {
        remoteStandsInTerm.forEach(s => allRemoteStands.push(s));
      }
    });
    if (allRemoteStands.length) {
      labelRows.push('<div class="alloc-gantt-section-spacer" aria-hidden="true"></div>');
      trackRows.push(
        '<div class="alloc-row" data-stand-id="">' +
          '<div class="alloc-row-track" data-stand-id="" style="background:transparent;border:none;height:8px;min-height:8px;"></div>' +
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
          '<div class="alloc-row-track" data-stand-id="" style="background:transparent;border:none;height:20px;min-height:20px;"></div>' +
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
    var allStands = (state.pbbStands || []).concat(state.remoteStands || []);
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
      if (typeof renderFlightList === 'function')
        renderFlightList(false, false, { scheduleMode: 'incremental', dirtyFlightIds: [ctx.flightId], touchedStandIds: touched });
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
        renderFlightList(false, false, { scheduleMode: 'incremental', dirtyFlightIds: [ctxFid], touchedStandIds: touched });
      }
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

  function validateNetworkForFlights() {
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
    const termsForLabel = makeUniqueNamedCopy(state.terminals || [], 'name').map(function(t) { return {
      id: t.id,
      name: (t.name || '').trim() || 'Building'
    }; });
    function termNameById(id) {
      const tt = termsForLabel.find(function(t) { return t.id === id; });
      return tt ? tt.name : (id || 'Building');
    }
    const allStands = (state.pbbStands || []).concat(state.remoteStands || []);
    (state.flights || []).forEach(function(f) {
      if (!f || !f.standId) return;
      const stand = allStands.find(function(s) { return s.id === f.standId; });
      if (!stand) return;
      const isRemote = (state.remoteStands || []).some(function(r) { return r.id === stand.id; });
      if (!isRemote) return;
      const termId = (f.token && f.token.terminalId) || null;
      if (!termId) return;
      const allowed = Array.isArray(stand.allowedTerminals) ? stand.allowedTerminals : [];
      if (allowed.length && !allowed.includes(termId)) {
        const flightLabel = f.id || f.flightNo || f.reg || '';
        const standLabel = stand.name || 'Remote';
        const termLabel = termNameById(termId);
        const allowedLabel = allowed.map(termNameById).join(', ');
        msgs.push('Flight ' + (flightLabel || '') + ' building setting(' + termLabel + ') does not match Remote stand ' + standLabel + ' available building settings (' + allowedLabel + ').');
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
    const stands = (state.pbbStands || []).concat(state.remoteStands || []);
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
  function kpiGateChartPlaceholder(buckets) {
    if (!buckets || !buckets.length) return '<div class="kpi-empty-state">No gate occupancy data is available for the current snapshot.</div>';
    return '<div class="kpi-chart-canvas-host kpi-chart-wrap--gate-fill"><canvas id="kpiChartGateOcc" aria-label="Gate occupancy chart"></canvas></div>';
  }
  function kpiRunwayChartPlaceholder(buckets) {
    if (!buckets || !buckets.length) return '<div class="kpi-empty-state">No arrival or departure events are available for the current snapshot.</div>';
    return '<div class="kpi-chart-canvas-host"><canvas id="kpiChartRunway" aria-label="Runway traffic chart"></canvas></div>';
  }

  function collectKpiSnapshot() {
    const flights = Array.isArray(state.flights) ? state.flights.slice() : [];
    const rows = flights.map(function(f) {
      const arrTaxiMin = kpiToNumber(typeof getBaseVttArrMinutes === 'function' ? getBaseVttArrMinutes(f) : null);
      const depBlockOutMin = kpiToNumber(typeof getDepBlockOutMin === 'function' ? getDepBlockOutMin(f) : null);
      const depTaxiMin = kpiToNumber(typeof getBaseVttDepMinutesToLineup === 'function' ? getBaseVttDepMinutesToLineup(f) : null);
      const rotSec = kpiToNumber(f && f.arrRotSec != null ? f.arrRotSec : (typeof getArrRotMinutes === 'function' ? getArrRotMinutes(f) * 60 : null));
      const depRotSec = (f && f.arrDep === 'Dep' && typeof computeDepRotSecondsForFlight === 'function')
        ? computeDepRotSecondsForFlight(f)
        : ((typeof SCHED_DEP_ROT_MIN === 'number' && isFinite(SCHED_DEP_ROT_MIN)) ? SCHED_DEP_ROT_MIN * 60 : null);
      const arrTaxiDelayMin = kpiToNumber(f && f.vttADelayMin != null ? f.vttADelayMin : 0);
      const depTaxiDelayMin = kpiToNumber(f && f.depTaxiDelayMin != null ? f.depTaxiDelayMin : 0);
      const sibt = kpiToNumber(f && f.sibtMin_orig != null ? f.sibtMin_orig : (f && f.timeMin != null ? f.timeMin : null));
      const sldt = kpiToNumber(f && f.sldtMin_orig != null ? f.sldtMin_orig : (sibt != null && arrTaxiMin != null && rotSec != null ? Math.max(0, sibt - arrTaxiMin - rotSec / 60) : null));
      const dwellMin = kpiToNumber(f && f.dwellMin != null ? f.dwellMin : null);
      const sobt = kpiToNumber(f && f.sobtMin_orig != null ? f.sobtMin_orig : (sibt != null && dwellMin != null ? sibt + dwellMin : null));
      const sttDepMinK = kpiToNumber(typeof getBaseVttDepMinutesToHoldingSlot === 'function' ? getBaseVttDepMinutesToHoldingSlot(f) : depTaxiMin);
      const depRotMinK = depRotSec != null && isFinite(depRotSec) ? depRotSec / 60 : null;
      const stot = kpiToNumber(f && f.stotMin_orig != null ? f.stotMin_orig : (sobt != null && depRotMinK != null && sttDepMinK != null ? sobt + depRotMinK + sttDepMinK : (sobt != null && depBlockOutMin != null ? sobt + depBlockOutMin : null)));
      const eldt = kpiToNumber(f && f.eldtMin != null ? f.eldtMin : (f && f.sldtMin_d != null ? f.sldtMin_d : sldt));
      const eibt = kpiToNumber(f && f.eibtMin != null ? f.eibtMin : (eldt != null && arrTaxiMin != null && rotSec != null ? eldt + arrTaxiMin + rotSec / 60 + (kpiToNumber(f.vttADelayMin) || 0) : sibt));
      const eobt = kpiToNumber(f && f.eobtMin != null ? f.eobtMin : sobt);
      const etot = kpiToNumber(f && f.etotMin != null ? f.etotMin : (f && f.stotMin_d != null ? f.stotMin_d : stot));
      const failed = !!(f && flightBlockedLikeNoWay(f));
      const paxArrDelay = (eibt != null && sibt != null) ? Math.max(0, eibt - sibt) : null;
      const paxDepDelay = (eobt != null && sobt != null) ? Math.max(0, eobt - sobt) : null;
      const acArrDelay = (eldt != null && sldt != null) ? Math.max(0, eldt - sldt) : null;
      const acDepDelay = (etot != null && stot != null) ? Math.max(0, etot - stot) : null;
      return {
        flight: f,
        id: f && f.id ? f.id : '',
        reg: f && f.reg ? f.reg : '',
        flightNumber: f && f.flightNumber ? f.flightNumber : '',
        standId: f && f.standId ? f.standId : null,
        standName: kpiStandLabelById(f && f.standId ? f.standId : null),
        arrTaxiMin,
        depTaxiMin,
        rotSec,
        depRotSec,
        arrTaxiDelayMin,
        depTaxiDelayMin,
        sibt,
        sobt,
        sldt,
        stot,
        eldt,
        eibt,
        eobt,
        etot,
        failed,
        paxArrDelay,
        paxDepDelay,
        acArrDelay,
        acDepDelay
      };
    });
    const KPI_ROLL_STEP_MIN = 15;
    const KPI_ROLL_WIN_MIN = 60;
    const buckets = [];
    if (rows.length) {
      const wLastStart = 1440 - KPI_ROLL_WIN_MIN;
      for (let w = 0; w <= wLastStart; w += KPI_ROLL_STEP_MIN) {
        const wPlus = w + KPI_ROLL_WIN_MIN;
        const activeStands = new Set();
        let arrivals = 0;
        let departures = 0;
        rows.forEach(function(row) {
          const occStartRaw = row.eibt != null ? row.eibt : row.sibt;
          const occEndRaw = row.eobt != null ? row.eobt : row.sobt;
          const osStart = kpiMinuteOfDay(occStartRaw);
          const osEnd = kpiMinuteOfDay(occEndRaw);
          if (row.standId && osStart != null && osEnd != null &&
              kpiRollWindowOverlapsInterval(w, KPI_ROLL_WIN_MIN, osStart, osEnd)) {
            activeStands.add(row.standId);
          }
          const eldtM = kpiMinuteOfDay(row.eldt);
          const etotM = kpiMinuteOfDay(row.etot);
          if (eldtM != null && eldtM >= w && eldtM < wPlus) arrivals += 1;
          if (etotM != null && etotM >= w && etotM < wPlus) departures += 1;
        });
        buckets.push({
          label: kpiFormatClockBucket15(w),
          occupancy: activeStands.size,
          arrivals: arrivals,
          departures: departures,
          total: arrivals + departures,
          bucketStart: w
        });
      }
    }
    const failedFlights = rows.filter(function(row) { return row.failed; });
    const operationalFlights = rows.filter(function(row) { return !row.failed; });
    const peakBucket = buckets.reduce(function(best, bucket) {
      if (!best) return bucket;
      return (bucket.occupancy || 0) > (best.occupancy || 0) ? bucket : best;
    }, null);
    const busiestBucket = buckets.reduce(function(best, bucket) {
      if (!best) return bucket;
      return (bucket.total || 0) > (best.total || 0) ? bucket : best;
    }, null);
    const peakRunwayArrBucket = buckets.reduce(function(best, bucket) {
      if (!best) return bucket;
      return (bucket.arrivals || 0) > (best.arrivals || 0) ? bucket : best;
    }, null);
    const peakRunwayDepBucket = buckets.reduce(function(best, bucket) {
      if (!best) return bucket;
      return (bucket.departures || 0) > (best.departures || 0) ? bucket : best;
    }, null);
    const detailRows = rows.slice().sort(function(a, b) {
      const delayA = (a.paxArrDelay || 0) + (a.paxDepDelay || 0) + (a.acArrDelay || 0) + (a.acDepDelay || 0);
      const delayB = (b.paxArrDelay || 0) + (b.paxDepDelay || 0) + (b.acArrDelay || 0) + (b.acDepDelay || 0);
      return delayB - delayA;
    });
    return {
      rows: rows,
      buckets: buckets,
      totalFlights: rows.length,
      failedFlights: failedFlights.length,
      operationalFlights: operationalFlights.length,
      peakBucket: peakBucket,
      busiestBucket: busiestBucket,
      peakRunwayArrBucket: peakRunwayArrBucket,
      peakRunwayDepBucket: peakRunwayDepBucket,
      rotArrTotalSec: kpiSum(rows, function(row) { return row.rotSec; }),
      rotArrAvgSec: kpiAverage(rows, function(row) { return row.rotSec; }),
      rotDepTotalSec: kpiSum(rows, function(row) { return row.depRotSec; }),
      rotDepAvgSec: kpiAverage(rows, function(row) { return row.depRotSec; }),
      arrTaxiTotalMin: kpiSum(rows, function(row) { return row.arrTaxiMin; }),
      arrTaxiAvgMin: kpiAverage(rows, function(row) { return row.arrTaxiMin; }),
      depTaxiTotalMin: kpiSum(rows, function(row) { return row.depTaxiMin; }),
      depTaxiAvgMin: kpiAverage(rows, function(row) { return row.depTaxiMin; }),
      arrTaxiDelayTotalMin: kpiSum(rows, function(row) { return row.arrTaxiDelayMin; }),
      arrTaxiDelayAvgMin: kpiAverage(rows, function(row) { return row.arrTaxiDelayMin; }),
      depTaxiDelayTotalMin: kpiSum(rows, function(row) { return row.depTaxiDelayMin; }),
      depTaxiDelayAvgMin: kpiAverage(rows, function(row) { return row.depTaxiDelayMin; }),
      paxArrDelayTotalMin: kpiSum(rows, function(row) { return row.paxArrDelay; }),
      paxArrDelayAvgMin: kpiAverage(rows, function(row) { return row.paxArrDelay; }),
      paxDepDelayTotalMin: kpiSum(rows, function(row) { return row.paxDepDelay; }),
      paxDepDelayAvgMin: kpiAverage(rows, function(row) { return row.paxDepDelay; }),
      acArrDelayTotalMin: kpiSum(rows, function(row) { return row.acArrDelay; }),
      acArrDelayAvgMin: kpiAverage(rows, function(row) { return row.acArrDelay; }),
      acDepDelayTotalMin: kpiSum(rows, function(row) { return row.acDepDelay; }),
      acDepDelayAvgMin: kpiAverage(rows, function(row) { return row.acDepDelay; }),
      detailRows: detailRows
    };
  }

  function renderKpiDashboard(reasonLabel) {
    const host = document.getElementById('kpiDashboard');
    const status = document.getElementById('kpiSnapshotStatus');
    if (!host) return;
    if (reasonLabel === 'Updated') state.kpiRollingDetailExpanded = false;
    if (!host._kpiRollingMoreBound) {
      host._kpiRollingMoreBound = true;
      host.addEventListener('click', function(ev) {
        const t = ev.target;
        if (t && t.id === 'btnKpiRollingExpand') {
          state.kpiRollingDetailExpanded = true;
          renderKpiDashboard('Expanded');
        }
      });
    }
    kpiDisposeInteractiveCharts();
    const snapshot = collectKpiSnapshot();
    if (!snapshot.totalFlights) {
      host.innerHTML = '<div class="kpi-empty-state">No flights are available yet. Add or load a schedule, then click <strong>Pro Sim</strong> to refresh the KPI snapshot.</div>';
      if (status) status.textContent = (reasonLabel || 'Snapshot') + ' · ' + kpiFormatSnapshotTime();
      return;
    }
    const prArr = snapshot.peakRunwayArrBucket;
    const prDep = snapshot.peakRunwayDepBucket;
    const pkOcc = snapshot.peakBucket;
    const peakRunwayArrText = prArr ? (kpiFormatCount(prArr.arrivals || 0) + ' · ' + prArr.label) : '—';
    const peakRunwayDepText = prDep ? (kpiFormatCount(prDep.departures || 0) + ' · ' + prDep.label) : '—';
    const peakGateText = pkOcc ? (kpiFormatCount(pkOcc.occupancy || 0) + ' · ' + pkOcc.label) : '—';
    const busiestText = snapshot.busiestBucket ? (kpiFormatCount(snapshot.busiestBucket.total) + ' · ' + snapshot.busiestBucket.label) : '—';
    const busiestMeta = snapshot.busiestBucket ? ('15m step · 60m rolling · ELDT+ETOT') : 'No runway data';
    const summaryCards = [
      kpiBuildSummaryCard('Total Flights', kpiFormatCount(snapshot.totalFlights), 'accent'),
      kpiBuildSummaryCard('Failed Flights', kpiFormatCount(snapshot.failedFlights), snapshot.failedFlights > 0 ? 'danger' : 'success'),
      kpiBuildSummaryCard('Peak Runway Arr', peakRunwayArrText, 'warning'),
      kpiBuildSummaryCard('Peak Runway Dep', peakRunwayDepText, 'warning'),
      kpiBuildSummaryCard('Peak Gate Occupancy', peakGateText, 'accent')
    ].join('');
    const panelHtml = [
      kpiBuildPanel('Surface Movement', 'ROT · Taxi · Taxi delay', [
        kpiBuildMetricRow('Arr ROT time', 'Avg ' + kpiFormatSecondsValue(snapshot.rotArrAvgSec), 'Total ' + kpiFormatSecondsValue(snapshot.rotArrTotalSec)),
        kpiBuildMetricRow('Dep ROT time', 'Avg ' + kpiFormatSecondsValue(snapshot.rotDepAvgSec), 'Total ' + kpiFormatSecondsValue(snapshot.rotDepTotalSec)),
        kpiBuildMetricRow('Arr taxi time', 'Avg ' + kpiFormatMinutesValue(snapshot.arrTaxiAvgMin), 'Total ' + kpiFormatMinutesValue(snapshot.arrTaxiTotalMin)),
        kpiBuildMetricRow('Dep taxi time', 'Avg ' + kpiFormatMinutesValue(snapshot.depTaxiAvgMin), 'Total ' + kpiFormatMinutesValue(snapshot.depTaxiTotalMin)),
        kpiBuildMetricRow('Arr taxi delay', 'Avg ' + kpiFormatMinutesValue(snapshot.arrTaxiDelayAvgMin), 'Total ' + kpiFormatMinutesValue(snapshot.arrTaxiDelayTotalMin)),
        kpiBuildMetricRow('Dep taxi delay', 'Avg ' + kpiFormatMinutesValue(snapshot.depTaxiDelayAvgMin), 'Total ' + kpiFormatMinutesValue(snapshot.depTaxiDelayTotalMin))
      ]),
      kpiBuildPanel('Gate Delay', 'EIBT/EOBT vs schedule', [
        kpiBuildMetricRow('EIBT − SIBT', 'Avg ' + kpiFormatMinutesValue(snapshot.paxArrDelayAvgMin), 'Total ' + kpiFormatMinutesValue(snapshot.paxArrDelayTotalMin)),
        kpiBuildMetricRow('EOBT − SOBT', 'Avg ' + kpiFormatMinutesValue(snapshot.paxDepDelayAvgMin), 'Total ' + kpiFormatMinutesValue(snapshot.paxDepDelayTotalMin)),
        kpiBuildMetricRow('Busiest runway window', busiestText, busiestMeta)
      ]),
      kpiBuildPanel('Runway Delay', 'ELDT/ETOT vs schedule', [
        kpiBuildMetricRow('ELDT − SLDT', 'Avg ' + kpiFormatMinutesValue(snapshot.acArrDelayAvgMin), 'Total ' + kpiFormatMinutesValue(snapshot.acArrDelayTotalMin)),
        kpiBuildMetricRow('ETOT − STOT', 'Avg ' + kpiFormatMinutesValue(snapshot.acDepDelayAvgMin), 'Total ' + kpiFormatMinutesValue(snapshot.acDepDelayTotalMin)),
        kpiBuildMetricRow('Snapshot basis', kpiFormatCount(snapshot.totalFlights) + ' flights', 'Rendered only on initial load and Pro Sim')
      ])
    ].join('');
    const bucketsAll = snapshot.buckets || [];
    const capRows = KPI_ROLLING_TABLE_VISIBLE_ROWS;
    const rollExpanded = !!state.kpiRollingDetailExpanded;
    const bucketsForTable = (!rollExpanded && bucketsAll.length > capRows) ? bucketsAll.slice(0, capRows) : bucketsAll;
    const hourlyTableRows = bucketsForTable.map(function(bucket) {
      const highlight = snapshot.peakBucket && bucket.bucketStart === snapshot.peakBucket.bucketStart ? ' class="kpi-row-highlight"' : '';
      return '' +
        '<tr' + highlight + '>' +
          '<td>' + escapeHtml(bucket.label) + '</td>' +
          '<td>' + escapeHtml(kpiFormatCount(bucket.occupancy)) + '</td>' +
          '<td>' + escapeHtml(kpiFormatCount(bucket.arrivals)) + '</td>' +
          '<td>' + escapeHtml(kpiFormatCount(bucket.departures)) + '</td>' +
          '<td>' + escapeHtml(kpiFormatCount(bucket.total)) + '</td>' +
        '</tr>';
    }).join('');
    const rollingMoreRow = (!rollExpanded && bucketsAll.length > capRows)
      ? ('<tr class="kpi-rolling-more"><td colspan="5" style="font-size:11px;color:#9ca3af;padding:8px 6px;">' +
          '<button type="button" class="tool-btn" id="btnKpiRollingExpand">더 보기 (' + String(bucketsAll.length - capRows) + '행)</button>' +
        '</td></tr>')
      : '';
    const topDelayRows = snapshot.detailRows.slice(0, 10).map(function(row) {
      const statusClass = row.failed ? 'fail' : 'ok';
      const statusLabel = row.failed ? 'Failed' : 'Normal';
      return '' +
        '<tr>' +
          '<td>' + escapeHtml((row.reg || row.flightNumber || row.id || '—')) + '</td>' +
          '<td>' + escapeHtml(row.standName || 'Unassigned') + '</td>' +
          '<td>' + escapeHtml(kpiFormatMinutesValue(row.paxArrDelay)) + '</td>' +
          '<td>' + escapeHtml(kpiFormatMinutesValue(row.paxDepDelay)) + '</td>' +
          '<td>' + escapeHtml(kpiFormatMinutesValue((row.acArrDelay || 0) + (row.acDepDelay || 0))) + '</td>' +
          '<td><span class="kpi-badge ' + statusClass + '">' + escapeHtml(statusLabel) + '</span></td>' +
        '</tr>';
    }).join('');
    host.innerHTML = '' +
      '<div class="kpi-summary-grid">' + summaryCards + '</div>' +
      '<div class="kpi-panel-grid">' + panelHtml + '</div>' +
      '<div class="kpi-chart-grid">' +
        '<div class="kpi-chart-card kpi-chart-card-primary">' +
          '<div class="kpi-chart-head">' +
            '<div>' +
              '<div class="kpi-chart-title">Hourly Gate Occupancy</div>' +


              '<div class="kpi-chart-subtitle">15m anchors · rolling 60m: unique stands overlapping EIBT–EOBT with [w, w+60).</div>' +
            '</div>' +
            '<div class="kpi-chart-legend">' +
              '<span class="kpi-legend-item"><span class="kpi-legend-swatch" style="background:#a78bfa;"></span>Gate occupancy</span>' +
            '</div>' +
          '</div>' +
          kpiGateChartPlaceholder(snapshot.buckets) +
        '</div>' +
        '<div class="kpi-chart-card kpi-chart-card-primary">' +
          '<div class="kpi-chart-head">' +
            '<div>' +
              '<div class="kpi-chart-title">Hourly Runway Traffic</div>' +
              '<div class="kpi-chart-subtitle">15m anchors · rolling 60m: ELDT arrivals and ETOT departures in [w, w+60).</div>' +
            '</div>' +
            '<div class="kpi-chart-legend">' +
              '<span class="kpi-legend-item"><span class="kpi-legend-swatch" style="background:#38bdf8;"></span>Arrivals</span>' +
              '<span class="kpi-legend-item"><span class="kpi-legend-swatch" style="background:#fb923c;"></span>Departures</span>' +
              '<span class="kpi-legend-item"><span class="kpi-legend-swatch" style="background:#c4b5fd;"></span>Total</span>' +
            '</div>' +
          '</div>' +
          kpiRunwayChartPlaceholder(snapshot.buckets) +
        '</div>' +
      '</div>' +
      '<div class="kpi-detail-grid">' +
        '<div class="kpi-table-card">' +
          '<div class="kpi-chart-title">Rolling window detail</div>' +
          '<div class="kpi-chart-subtitle">Same 15m / 60m windows: gate occupancy; runway arr/dep = ELDT / ETOT counts.</div>' +
          '<div class="kpi-table-wrap">' +
            '<table class="kpi-table">' +
              '<thead><tr><th>Window w</th><th>Gate occ</th><th>Runway arr</th><th>Runway dep</th><th>Total</th></tr></thead>' +
              '<tbody>' + hourlyTableRows + rollingMoreRow + '</tbody>' +
            '</table>' +
          '</div>' +
        '</div>' +
        '<div class="kpi-table-card">' +
          '<div class="kpi-chart-title">Top Delay Flights</div>' +
          '<div class="kpi-chart-subtitle">Largest combined gate delay (EIBT/SIBT, EOBT/SOBT) and runway delay (ELDT/SLDT, ETOT/STOT) footprint.</div>' +
          '<div class="kpi-table-wrap">' +
            '<table class="kpi-table">' +
              '<thead><tr><th>Flight</th><th>Stand</th><th>Gate Arr Delay</th><th>Gate Dep Delay</th><th>Runway Delay</th><th>Status</th></tr></thead>' +
              '<tbody>' + topDelayRows + '</tbody>' +
            '</table>' +
          '</div>' +
        '</div>' +
      '</div>';
    if (status) status.textContent = (reasonLabel || 'Snapshot') + ' · ' + kpiFormatSnapshotTime();
    kpiMountInteractiveCharts(snapshot.buckets || []);
  }

  function scheduledSldtFromSibtMinutes(f, sibtMin) {
    const sibt = sibtMin != null && isFinite(sibtMin) ? sibtMin : 0;
    const vttArrMin = getBaseVttArrMinutes(f);
    const rotArrMin = getArrRotMinutes(f);
    return Math.max(0, sibt - vttArrMin - rotArrMin);
  }
  function scheduledStotFromSobtMinutes(f, sobtMin) {
    const sobt = sobtMin != null && isFinite(sobtMin) ? sobtMin : 0;
    const depRotSec = (typeof computeDepRotSecondsForFlight === 'function')
      ? computeDepRotSecondsForFlight(f)
      : Math.max(0, Number(SCHED_DEP_ROT_MIN) || 2) * 60;
    const rotDepMin = depRotSec / 60;
    const depBlockOutMin = (typeof getDepBlockOutMin === 'function') ? getDepBlockOutMin(f) : 0;
    const rollBundleSecFallback = DEP_LINEUP_HOLD_SEC + takeoffRollSecForRunwayTailLenM(0, DEP_TAKEOFF_ACCEL_SMALL_MS2);
    const vttDepMinLineup = (typeof getBaseVttDepMinutesToLineup === 'function')
      ? getBaseVttDepMinutesToLineup(f)
      : Math.max(0, depBlockOutMin - ((typeof computeDepRollAndLineupOnlySec === 'function') ? computeDepRollAndLineupOnlySec(f) : rollBundleSecFallback) / 60);
    const sttDepMin = (typeof getBaseVttDepMinutesToHoldingSlot === 'function') ? getBaseVttDepMinutesToHoldingSlot(f) : vttDepMinLineup;
    return sobt + rotDepMin + sttDepMin;
  }
  function scheduledSobtFromStotMinutes(f, stotMin) {
    const stot = stotMin != null && isFinite(stotMin) ? stotMin : 0;
    const depRotSec = (typeof computeDepRotSecondsForFlight === 'function')
      ? computeDepRotSecondsForFlight(f)
      : Math.max(0, Number(SCHED_DEP_ROT_MIN) || 2) * 60;
    const rotDepMin = depRotSec / 60;
    const depBlockOutMin = (typeof getDepBlockOutMin === 'function') ? getDepBlockOutMin(f) : 0;
    const rollBundleSecFallback = DEP_LINEUP_HOLD_SEC + takeoffRollSecForRunwayTailLenM(0, DEP_TAKEOFF_ACCEL_SMALL_MS2);
    const vttDepMinLineup = (typeof getBaseVttDepMinutesToLineup === 'function')
      ? getBaseVttDepMinutesToLineup(f)
      : Math.max(0, depBlockOutMin - ((typeof computeDepRollAndLineupOnlySec === 'function') ? computeDepRollAndLineupOnlySec(f) : rollBundleSecFallback) / 60);
    const sttDepMin = (typeof getBaseVttDepMinutesToHoldingSlot === 'function') ? getBaseVttDepMinutesToHoldingSlot(f) : vttDepMinLineup;
    return Math.max(0, stot - rotDepMin - sttDepMin);
  }
  function applyScheduledGateTimingFromSField(f, field, minutes) {
    if (!f || flightBlockedLikeNoWay(f)) return false;
    const m = Number(minutes);
    if (!isFinite(m) || m < 0) return false;
    let dwell = f.dwellMin != null ? f.dwellMin : 0;
    let minDwell = f.minDwellMin != null ? f.minDwellMin : 0;
    dwell = Math.max(SCHED_DWELL_FLOOR_MIN, dwell);
    minDwell = Math.max(SCHED_DWELL_FLOOR_MIN, minDwell);
    if (minDwell > dwell) minDwell = dwell;
    if (field === 'sldt') {
      const vttArrMin = getBaseVttArrMinutes(f);
      const rotArrMin = getArrRotMinutes(f);
      f.sldtMin_orig = m;
      const sibt = Math.max(0, m + vttArrMin + rotArrMin);
      f.timeMin = sibt;
      f.sibtMin_orig = sibt;
      f.sobtMin_orig = sibt + dwell;
      f.stotMin_orig = scheduledStotFromSobtMinutes(f, f.sobtMin_orig);
      f.dwellMin = dwell;
      f.minDwellMin = minDwell;
      return true;
    }
    if (field === 'sibt') {
      f.timeMin = m;
      f.sibtMin_orig = m;
      f.sldtMin_orig = scheduledSldtFromSibtMinutes(f, m);
      f.sobtMin_orig = m + dwell;
      f.stotMin_orig = scheduledStotFromSobtMinutes(f, f.sobtMin_orig);
      f.dwellMin = dwell;
      f.minDwellMin = minDwell;
      return true;
    }
    if (field === 'sobt') {
      const sibt = f.timeMin != null ? f.timeMin : 0;
      let sobtAdj = Math.max(m, sibt + minDwell);
      f.sobtMin_orig = sobtAdj;
      f.dwellMin = Math.max(SCHED_DWELL_FLOOR_MIN, sobtAdj - sibt);
      if (f.minDwellMin != null) {
        f.minDwellMin = Math.max(SCHED_DWELL_FLOOR_MIN, Math.min(f.dwellMin, f.minDwellMin));
      }
      f.stotMin_orig = scheduledStotFromSobtMinutes(f, f.sobtMin_orig);
      return true;
    }
    if (field === 'stot') {
      const sibt = f.timeMin != null ? f.timeMin : 0;
      const sobtGuess = scheduledSobtFromStotMinutes(f, m);
      let sobtAdj = Math.max(sobtGuess, sibt + minDwell);
      f.sobtMin_orig = sobtAdj;
      f.dwellMin = Math.max(SCHED_DWELL_FLOOR_MIN, sobtAdj - sibt);
      if (f.minDwellMin != null) {
        f.minDwellMin = Math.max(SCHED_DWELL_FLOOR_MIN, Math.min(f.dwellMin, f.minDwellMin));
      }
      f.stotMin_orig = scheduledStotFromSobtMinutes(f, f.sobtMin_orig);
      return true;
    }
    return false;
  }

  function applySdDispDeltaFromSibtSobt(f) {
    if (!f || flightBlockedLikeNoWay(f)) return;
    const sibt = f.sibtMin_d;
    const sobt = f.sobtMin_d;
    if (typeof sibt === 'number' && isFinite(sibt)) {
      f.sldtMin_d = Math.max(0, sibt - SCHED_SD_SIBT_MINUS_SLD_MIN);
    }
    if (typeof sobt === 'number' && isFinite(sobt)) {
      f.stotMin_d = sobt + SCHED_SD_STOT_PLUS_SOBD_MIN;
    }
  }

  function computeScheduledDisplayTimes(flights) {
    if (!flights || !flights.length) return;
    flights.forEach(f => {
      if (flightBlockedLikeNoWay(f)) return;
      f.vttADelayMin = 0;
      const tArrMin = f.timeMin != null ? f.timeMin : 0;
      let dwell = f.dwellMin != null ? f.dwellMin : 0;
      let minDwell = f.minDwellMin != null ? f.minDwellMin : 0;
      dwell = Math.max(SCHED_DWELL_FLOOR_MIN, dwell);
      minDwell = Math.max(SCHED_DWELL_FLOOR_MIN, minDwell);
      if (minDwell > dwell) minDwell = dwell;
      f.dwellMin = dwell;
      f.minDwellMin = minDwell;
      const sldtOrig = scheduledSldtFromSibtMinutes(f, tArrMin);
      const sobtOrig = tArrMin + dwell;
      const stotOrig = scheduledStotFromSobtMinutes(f, sobtOrig);
      f.sldtMin_orig = sldtOrig;
      f.sibtMin_orig = tArrMin;
      f.sobtMin_orig = sobtOrig;
      f.stotMin_orig = stotOrig;
      f.sibtMin_d = tArrMin;
      f.sobtMin_d = sobtOrig;
      applySdDispDeltaFromSibtSobt(f);
    });
    const standToFlights = {};
    flights.forEach(f => {
      if (flightBlockedLikeNoWay(f) || !f.standId) return;
      const sid = f.standId;
      if (!standToFlights[sid]) standToFlights[sid] = [];
      standToFlights[sid].push(f);
    });
    Object.keys(standToFlights).forEach(standId => {
      const list = standToFlights[standId];
      list.sort((a, b) => (a.sibtMin_d != null ? a.sibtMin_d : 0) - (b.sibtMin_d != null ? b.sibtMin_d : 0));
      let prevSOBT = -1e9;
      list.forEach(f => {
        const sibt0 = (f.sibtMin_d != null ? f.sibtMin_d : 0);
        const overlap = Math.max(0, prevSOBT - sibt0);
        f.vttADelayMin = overlap;
        f.sibtMin_d = sibt0 + overlap;
        const dwell = f.dwellMin != null ? f.dwellMin : SCHED_DWELL_FLOOR_MIN;
        const minDwell = f.minDwellMin != null ? f.minDwellMin : SCHED_DWELL_FLOOR_MIN;
        const minSobtByDwell = f.sibtMin_d + minDwell;
        const sobtCandidate = (f.sobtMin_d != null ? f.sobtMin_d : (f.sibtMin_d + dwell));
        f.sobtMin_d = Math.max(sobtCandidate, minSobtByDwell);
        applySdDispDeltaFromSibtSobt(f);
        prevSOBT = f.sobtMin_d;
      });
    });
    flights.forEach(f => {
      if (!f || flightBlockedLikeNoWay(f) || !f.standId) return;
      const dwell = f.dwellMin != null ? f.dwellMin : SCHED_DWELL_FLOOR_MIN;
      const minDwell = f.minDwellMin != null ? f.minDwellMin : SCHED_DWELL_FLOOR_MIN;
      const sibt = (f.sibtMin_d != null ? f.sibtMin_d
                   : (f.sibtMin_orig != null ? f.sibtMin_orig : 0));
      const minSobtByDwell = sibt + minDwell;
      const sobtCurrent = (f.sobtMin_d != null ? f.sobtMin_d : (sibt + dwell));
      if (sobtCurrent < minSobtByDwell) {
        f.sobtMin_d = minSobtByDwell;
        applySdDispDeltaFromSibtSobt(f);
      }
    });
    flights.forEach(f => {
      if (flightBlockedLikeNoWay(f)) return;
      f.sldtMin = f.sldtMin_d;
      f.stotMin = f.stotMin_d;
      f.sobtMin = f.sobtMin_d;
    });
  }

  function computeScheduledDisplayTimesIncremental(allFlights, dirtyFlightIds, touchedStandIds) {
    if (!allFlights || !allFlights.length) return;
    const dirty = (dirtyFlightIds instanceof Set) ? dirtyFlightIds : new Set(dirtyFlightIds || []);
    const touchedStands = (touchedStandIds instanceof Set) ? touchedStandIds : new Set(touchedStandIds || []);
    const standsToRecompute = new Set();
    touchedStands.forEach(function(sid) { if (sid != null && sid !== '') standsToRecompute.add(sid); });
    const needStep1 = new Set();
    dirty.forEach(function(id) { if (id != null && id !== '') needStep1.add(id); });
    allFlights.forEach(function(f) {
      if (!f || flightBlockedLikeNoWay(f)) return;
      if (f.standId && standsToRecompute.has(f.standId)) needStep1.add(f.id);
    });
    allFlights.forEach(function(f) {
      if (!f || !needStep1.has(f.id)) return;
      if (flightBlockedLikeNoWay(f)) return;
      f.vttADelayMin = 0;
      const tArrMin = f.timeMin != null ? f.timeMin : 0;
      let dwell = f.dwellMin != null ? f.dwellMin : 0;
      let minDwell = f.minDwellMin != null ? f.minDwellMin : 0;
      dwell = Math.max(SCHED_DWELL_FLOOR_MIN, dwell);
      minDwell = Math.max(SCHED_DWELL_FLOOR_MIN, minDwell);
      if (minDwell > dwell) minDwell = dwell;
      f.dwellMin = dwell;
      f.minDwellMin = minDwell;
      const sldtOrig = scheduledSldtFromSibtMinutes(f, tArrMin);
      const sobtOrig = tArrMin + dwell;
      const stotOrig = scheduledStotFromSobtMinutes(f, sobtOrig);
      f.sldtMin_orig = sldtOrig;
      f.sibtMin_orig = tArrMin;
      f.sobtMin_orig = sobtOrig;
      f.stotMin_orig = stotOrig;
      f.sibtMin_d = tArrMin;
      f.sobtMin_d = sobtOrig;
      applySdDispDeltaFromSibtSobt(f);
    });
    standsToRecompute.forEach(function(standId) {
      const list = allFlights.filter(function(f) {
        return f && !flightBlockedLikeNoWay(f) && f.standId === standId;
      });
      list.sort((a, b) => (a.sibtMin_d != null ? a.sibtMin_d : 0) - (b.sibtMin_d != null ? b.sibtMin_d : 0));
      let prevSOBT = -1e9;
      list.forEach(function(f) {
        const sibt0 = (f.sibtMin_d != null ? f.sibtMin_d : 0);
        const overlap = Math.max(0, prevSOBT - sibt0);
        f.vttADelayMin = overlap;
        f.sibtMin_d = sibt0 + overlap;
        const dwell = f.dwellMin != null ? f.dwellMin : SCHED_DWELL_FLOOR_MIN;
        const minDwell = f.minDwellMin != null ? f.minDwellMin : SCHED_DWELL_FLOOR_MIN;
        const minSobtByDwell = f.sibtMin_d + minDwell;
        const sobtCandidate = (f.sobtMin_d != null ? f.sobtMin_d : (f.sibtMin_d + dwell));
        f.sobtMin_d = Math.max(sobtCandidate, minSobtByDwell);
        applySdDispDeltaFromSibtSobt(f);
        prevSOBT = f.sobtMin_d;
      });
    });
    allFlights.forEach(function(f) {
      if (!f || flightBlockedLikeNoWay(f) || !f.standId) return;
      if (!standsToRecompute.has(f.standId)) return;
      const dwell = f.dwellMin != null ? f.dwellMin : SCHED_DWELL_FLOOR_MIN;
      const minDwell = f.minDwellMin != null ? f.minDwellMin : SCHED_DWELL_FLOOR_MIN;
      const sibt = (f.sibtMin_d != null ? f.sibtMin_d : (f.sibtMin_orig != null ? f.sibtMin_orig : 0));
      const minSobtByDwell = sibt + minDwell;
      const sobtCurrent = (f.sobtMin_d != null ? f.sobtMin_d : (sibt + dwell));
      if (sobtCurrent < minSobtByDwell) {
        f.sobtMin_d = minSobtByDwell;
        applySdDispDeltaFromSibtSobt(f);
      }
    });
    allFlights.forEach(function(f) {
      if (!f || flightBlockedLikeNoWay(f)) return;
      const onTouched = f.standId && standsToRecompute.has(f.standId);
      if (!needStep1.has(f.id) && !onTouched) return;
      f.sldtMin = f.sldtMin_d;
      f.stotMin = f.stotMin_d;
      f.sobtMin = f.sobtMin_d;
    });
  }

  function rsepGetSec(val) {
    const n = Number(val);
    return isFinite(n) && n >= 0 ? n : RSEP_MISSING_MATRIX_SEC;
  }

  function rsepApplySeparationToEvents(events, cfg) {
    const arrArr = (cfg.seqData && cfg.seqData['ARR→ARR']) ? cfg.seqData['ARR→ARR'] : {};
    const depDep = (cfg.seqData && cfg.seqData['DEP→DEP']) ? cfg.seqData['DEP→DEP'] : {};
    const depArr = (cfg.seqData && cfg.seqData['DEP→ARR']) ? cfg.seqData['DEP→ARR'] : {};
    const rot = (cfg.rot) ? cfg.rot : {};
    const getSec = rsepGetSec;
    events.sort((a, b) => a.time - b.time || a.index - b.index);
    let lastArrETime = -1e9, lastArrCat = null;
    let lastDepETime = -1e9, lastDepCat = null;
    events.forEach(ev => {
      if (ev.type === 'arr') {
        let minFromArr = lastArrETime >= -1e8 && lastArrCat ? lastArrETime + getSec((arrArr[lastArrCat] && arrArr[lastArrCat][ev.cat]) != null ? arrArr[lastArrCat][ev.cat] : RSEP_MISSING_MATRIX_SEC) / 60 : -1e9;
        let minFromDep = lastDepETime >= -1e8 && lastDepCat ? lastDepETime + getSec(depArr[ev.cat]) / 60 : -1e9;
        const eTime = Math.max(ev.time, minFromArr, minFromDep);
        ev.flight.eldtMin = eTime;
        lastArrETime = eTime;
        lastArrCat = ev.cat;
      } else {
        let minFromArr = lastArrETime >= -1e8 && lastArrCat ? lastArrETime + getSec(rot[lastArrCat]) / 60 : -1e9;
        let minFromDep = lastDepETime >= -1e8 && lastDepCat ? lastDepETime + getSec((depDep[lastDepCat] && depDep[lastDepCat][ev.cat]) != null ? depDep[lastDepCat][ev.cat] : RSEP_MISSING_MATRIX_SEC) / 60 : -1e9;
        const etotSep = Math.max(ev.time, minFromArr, minFromDep);
        const vttADelay = ev.flight.vttADelayMin != null ? ev.flight.vttADelayMin : 0;
        const rotM = (ev.rotArrMin != null && isFinite(ev.rotArrMin)) ? ev.rotArrMin : getArrRotMinutes(ev.flight);
        const eibtMin = (ev.flight.eldtMin != null ? ev.flight.eldtMin : 0) + rotM + (ev.vttArrMin || 0) + vttADelay;
        const vttDep = ev.vttDepMin || 0;
        const etotMin = etotSep;
        const eobtMin = etotMin - vttDep;
        ev.flight.etotMin = etotMin;
        lastDepETime = etotMin;
        lastDepCat = ev.cat;
      }
    });
    let minT = Infinity, maxT = -Infinity;
    events.forEach(ev => {
      const s = ev.time;
      const e = ev.type === 'arr'
        ? (ev.flight && ev.flight.eldtMin != null ? ev.flight.eldtMin : s)
        : (ev.flight && ev.flight.etotMin != null ? ev.flight.etotMin : s);
      if (s < minT) minT = s;
      if (e < minT) minT = e;
      if (s > maxT) maxT = s;
      if (e > maxT) maxT = e;
    });
    if (!isFinite(minT) || !isFinite(maxT)) { minT = 0; maxT = 60; } else if (maxT <= minT) { maxT = minT + 60; }
    return { minT, maxT };
  }

  function rsepCollectEventsForRunway(rwy, flights, runways) {
    const cfg = rsepGetConfigForRunway(rwy);
    if (!cfg) return null;
    const stdKey = cfg.standard || 'ICAO';
    const events = [];
    let eventIndex = 0;
    flights.forEach((f, flightIdx) => {
      if (flightBlockedLikeNoWay(f)) return;
      let arrRwy = f.arrRunwayId || (f.token && f.token.runwayId);
      let depRwy = f.depRunwayId || (f.token && f.token.depRunwayId);
      if (arrRwy == null && depRwy == null && runways.length === 1) { arrRwy = rwy.id; depRwy = rwy.id; }
      else if (depRwy == null && arrRwy === rwy.id) depRwy = rwy.id;
      else if (arrRwy == null && depRwy === rwy.id) arrRwy = rwy.id;
      if (arrRwy !== rwy.id && depRwy !== rwy.id) return;
      const ac = typeof getAircraftInfoByType === 'function' ? getAircraftInfoByType(f.aircraftType) : null;
      const cat = stdKey === 'ICAO' ? (ac && ac.icaoJHL ? ac.icaoJHL : 'M') : (ac && ac.recatEu ? ac.recatEu : 'D');
      const sldtMin_d = f.sldtMin_d != null ? f.sldtMin_d : 0;
      const stotMin_d = f.stotMin_d != null ? f.stotMin_d : 0;
      const sobtMin_d = f.sobtMin_d != null ? f.sobtMin_d : 0;
      const vttArrMin = getBaseVttArrMinutes(f);
      const rotArrMin = getArrRotMinutes(f);
      const vttDepMin = (typeof getDepBlockOutMin === 'function') ? getDepBlockOutMin(f) : 0;
      if (arrRwy === rwy.id) events.push({ time: sldtMin_d, type: 'arr', flight: f, cat: cat, vttArrMin, rotArrMin, index: eventIndex++ });
      if (depRwy === rwy.id) {
        events.push({ time: stotMin_d, type: 'dep', flight: f, cat: cat, vttDepMin, vttArrMin, rotArrMin, sobtMin: sobtMin_d, index: eventIndex++ });
      }
    });
    return { cfg, events };
  }

  function runSeparationPass(runways, flights, byRunway, phase) {
    if (phase === 'initial') {
      runways.forEach(rwy => {
        const pack = rsepCollectEventsForRunway(rwy, flights, runways);
        if (!pack) return;
        const { cfg, events } = pack;
        if (!events.length) {
          byRunway[rwy.id] = { events: [], minT: 0, maxT: 0 };
          return;
        }
        const { minT, maxT } = rsepApplySeparationToEvents(events, cfg);
        byRunway[rwy.id] = { events, minT, maxT };
      });
    } else {
      runways.forEach(rwy => {
        const cfg = rsepGetConfigForRunway(rwy);
        if (!cfg) return;
        const data = byRunway[rwy.id];
        if (!data || !data.events || !data.events.length) return;
        const events = data.events;
        events.forEach(ev => {
          ev.time = ev.type === 'arr'
            ? (ev.flight.eldtMin != null ? ev.flight.eldtMin : ev.time)
            : (ev.flight.etotMin != null ? ev.flight.etotMin : ev.time);
        });
        const { minT, maxT } = rsepApplySeparationToEvents(events, cfg);
        byRunway[rwy.id] = { events, minT, maxT };
      });
    }
  }

  function buildRunwaySeparationTimelineByRunwaySnapshot(flights) {
    const snapGen = state.rwySepSnapshotStaleGen | 0;
    if (state.__rwySepSnapCacheGen === snapGen && state.__rwySepSnapCache) return state.__rwySepSnapCache;
    const list = flights || state.flights || [];
    const runwaysRaw = (state.taxiways || []).filter(t => t.pathType === 'runway');
    if (!runwaysRaw.length) {
      state.__rwySepSnapCache = {};
      state.__rwySepSnapCacheGen = snapGen;
      return state.__rwySepSnapCache;
    }
    const runways = (function() {
      const idToIndex = {};
      runwaysRaw.forEach((r, i) => { if (r && r.id != null) idToIndex[r.id] = i; });
      const n = runwaysRaw.length;
      const indeg = new Array(n).fill(0);
      const adj = new Array(n).fill(0).map(() => []);
      list.forEach(f => {
        if (!f) return;
        let arrRwy = f.arrRunwayId || (f.token && f.token.runwayId);
        let depRwy = f.depRunwayId || (f.token && f.token.depRunwayId);
        if (!arrRwy || !depRwy || arrRwy === depRwy) return;
        const ai = idToIndex[arrRwy];
        const di = idToIndex[depRwy];
        if (ai == null || di == null) return;
        adj[ai].push(di);
        indeg[di] += 1;
      });
      const q = [];
      for (let i = 0; i < n; i++) if (indeg[i] === 0) q.push(i);
      const orderIdx = [];
      while (q.length) {
        const i = q.shift();
        orderIdx.push(i);
        adj[i].forEach(j => {
          indeg[j] -= 1;
          if (indeg[j] === 0) q.push(j);
        });
      }
      if (orderIdx.length !== n) return runwaysRaw;
      return orderIdx.map(i => runwaysRaw[i]);
    })();
    const byRunway = {};
    runways.forEach(rwy => {
      const pack = rsepCollectEventsForRunway(rwy, list, runways);
      if (!pack || !pack.events.length) {
        byRunway[rwy.id] = { events: [], minT: 0, maxT: 0 };
        return;
      }
      const events = pack.events.slice().sort((a, b) => a.time - b.time || a.index - b.index);
      let minT = Infinity, maxT = -Infinity;
      events.forEach(ev => {
        const s = ev.time;
        const f = ev.flight;
        const e = ev.type === 'arr'
          ? (f && f.eldtMin != null && isFinite(f.eldtMin) ? f.eldtMin : s)
          : (f && f.etotMin != null && isFinite(f.etotMin) ? f.etotMin : s);
        if (s < minT) minT = s;
        if (e < minT) minT = e;
        if (s > maxT) maxT = s;
        if (e > maxT) maxT = e;
      });
      if (!isFinite(minT) || !isFinite(maxT)) { minT = 0; maxT = 60; } else if (maxT <= minT) maxT = minT + 60;
      byRunway[rwy.id] = { events, minT, maxT };


    });
    state.__rwySepSnapCache = byRunway;
    state.__rwySepSnapCacheGen = snapGen;
    return byRunway;
  }

  function computeSeparationAdjustedTimes() {
    return {};
  }

  function getRunwayPath(runwayId) {
    const taxiways = state.taxiways || [];
    let rw = runwayId ? taxiways.find(t => t.id === runwayId && t.pathType === 'runway' && t.vertices && t.vertices.length >= 2) : null;
    if (!rw) rw = taxiways.find(t => t.pathType === 'runway' && t.vertices && t.vertices.length >= 2);
    if (!rw || !rw.vertices.length) return null;
    const pts = rw.vertices.map(v => cellToPixel(v.col, v.row));
    return { startPx: pts[0], endPx: pts[pts.length-1], pts };
  }

  function getRunwayPointAtDistance(runwayId, distM) {
    const path = getRunwayPath(runwayId);
    if (!path || !path.pts || path.pts.length < 2) return null;
    const pts = path.pts;
    let acc = 0;
    for (let i = 0; i < pts.length - 1; i++) {
      const p1 = pts[i];
      const p2 = pts[i + 1];
      const segLen = pathDist(p1, p2);
      if (!(segLen > 1e-6)) continue;
      if (acc + segLen >= distM) {
        const t = Math.max(0, Math.min(1, (distM - acc) / segLen));
        return [
          p1[0] + (p2[0] - p1[0]) * t,
          p1[1] + (p2[1] - p1[1]) * t
        ];
      }
      acc += segLen;
    }
    return pts[pts.length - 1];
  }

  function flightEMinutesPrefer(f, keys, fallback) {
    for (let ki = 0; ki < keys.length; ki++) {
      const v = f[keys[ki]];
      if (typeof v === 'number' && isFinite(v)) return v;
    }
    return fallback;
  }
  function touchdownDistMForTimeline(f) {
    if (typeof f.arrTdDistM === 'number' && isFinite(f.arrTdDistM) && f.arrTdDistM >= 0) return f.arrTdDistM;
    const ac = (typeof getAircraftInfoByType === 'function') ? getAircraftInfoByType(f.aircraftType) : null;
    const z = ac && typeof ac.touchdown_zone_avg_m === 'number' ? ac.touchdown_zone_avg_m : null;
    if (typeof z === 'number' && z > 0) return z;
    return 400;
  }
  function touchdownSpeedMsForTimeline(f) {
    let v = f.arrVTdMs;
    if (typeof v === 'number' && isFinite(v) && v > 0) return Math.max(1, v);
    const ac = (typeof getAircraftInfoByType === 'function') ? getAircraftInfoByType(f.aircraftType) : null;
    v = ac && typeof ac.touchdown_speed_avg_ms === 'number' ? ac.touchdown_speed_avg_ms : 70;
    return Math.max(1, v);
  }
  
  function getRunwayInboundUxyAtDistance(runwayId, rwDir, distAlong) {
    const r = getRunwayPath(runwayId);
    const anchor = getRunwayPointAtDistance(runwayId, distAlong);
    if (!r || !r.pts || r.pts.length < 2 || !anchor) return null;
    const pts = r.pts;
    let segIdx = Math.max(0, pts.length - 2);
    let acc = 0;
    for (let i = 0; i < pts.length - 1; i++) {
      const segLen = pathDist(pts[i], pts[i + 1]);
      if (segLen < 1e-9) continue;
      if (acc + segLen >= distAlong - 1e-6) { segIdx = i; break; }
      acc += segLen;
    }
    const p1 = pts[segIdx], p2 = pts[segIdx + 1];
    const segLen = pathDist(p1, p2) || 1;
    let ux = (p2[0] - p1[0]) / segLen, uy = (p2[1] - p1[1]) / segLen;
    if (rwDir === 'counter_clockwise') { ux = -ux; uy = -uy; }
    return { td: anchor, ux, uy };
  }
  
  function buildStraightApproachPolylineWorld(runwayId, rwDir, anchorDistAlong, totalM) {
    const ax = getRunwayInboundUxyAtDistance(runwayId, rwDir, anchorDistAlong);
    if (!ax) return null;
    const td = ax.td, ux = ax.ux, uy = ax.uy;
    const tm = Math.max(0, Number(totalM) || 0);
    const tdxy = [td[0], td[1]];
    if (tm < 1e-6) return { pts: [tdxy, tdxy], pathLen: 0 };
    const outer = [td[0] - ux * tm, td[1] - uy * tm];
    return { pts: [outer, tdxy], pathLen: pathDist(outer, tdxy) };
  }
  
  function arrivalApproachAnchorDistM(runwayId, tdDistAlong) {
    let anchorDist = runwayApproachThresholdDistAlongM(runwayId, tdDistAlong);
    if (!(typeof anchorDist === 'number' && isFinite(anchorDist) && anchorDist >= 0)) anchorDist = tdDistAlong;
    else if (anchorDist > tdDistAlong + 1e-3) anchorDist = tdDistAlong;
    return anchorDist;
  }
  function buildArrivalApproachPolylinePts(runwayId, rwDir, anchorDist, offset, tdPt) {
    const pack = buildStraightApproachPolylineWorld(runwayId, rwDir, anchorDist, offset);
    let apprPts;
    if (pack && pack.pts && pack.pts.length >= 2) {
      apprPts = pack.pts.slice();
      const lastAp = apprPts[apprPts.length - 1];
      if (Math.hypot(lastAp[0] - tdPt[0], lastAp[1] - tdPt[1]) > 1e-3) apprPts.push([tdPt[0], tdPt[1]]);
    } else {
      const rsPt = getRunwayPointAtDistance(runwayId, anchorDist);
      const outer = approachPointBeforeThresholdJs(runwayId, rwDir, offset, anchorDist);
      const mid = rsPt ? [rsPt[0], rsPt[1]] : [tdPt[0], tdPt[1]];
      apprPts = [outer, mid];
      if (rsPt && Math.hypot(rsPt[0] - tdPt[0], rsPt[1] - tdPt[1]) > 1e-3) apprPts.push([tdPt[0], tdPt[1]]);
    }
    return { pack: pack, apprPts: apprPts };
  }
  function arrivalApproachDurationSecBeforeEldt(f) {
    const vTd = Math.max(1, touchdownSpeedMsForTimeline(f));
    const token = f.token || {};
    const runwayId = f.arrRunwayIdUsed || token.arrRunwayId || token.runwayId || f.arrRunwayId;
    if (runwayId == null || runwayId === '') return APPROACH_OFFSET_WORLD_M / vTd;
    const rwDir = String(f.arrRunwayDirUsed || 'clockwise');
    const tdDist = touchdownDistMForTimeline(f);
    const anchorDist = arrivalApproachAnchorDistM(runwayId, tdDist);
    const tdPt = getRunwayPointAtDistance(runwayId, tdDist);
    if (!tdPt) return APPROACH_OFFSET_WORLD_M / vTd;
    const built = buildArrivalApproachPolylinePts(runwayId, rwDir, anchorDist, APPROACH_OFFSET_WORLD_M, tdPt);
    const apprPts = built.apprPts;
    if (!apprPts || apprPts.length < 2) return APPROACH_OFFSET_WORLD_M / vTd;
    return polylineRawDurationSegmentVelocities(apprPts, function() { return vTd; });
  }
  
  function getFlightAirsideWindowSec(f) {
    if (!f) return null;
    if (f.noWayArr && f.noWayDep) return null;
    if (f.arrDep === 'Dep') {
      const eobtMin = flightEMinutesPrefer(f, ['eobtMin'], flightEMinutesPrefer(f, ['timeMin'], 0) + (typeof f.dwellMin === 'number' ? f.dwellMin : 0));
      const etotMin = flightEMinutesPrefer(f, ['etotMin'], eobtMin + 30);
      const eobtS = eobtMin * 60;
      const etotS = etotMin * 60;
      const depRotS = Math.max(0, (typeof computeDepRotSecondsForFlight === 'function')
        ? computeDepRotSecondsForFlight(f)
        : (Math.max(0, Number(SCHED_DEP_ROT_MIN) || 0) * 60));
      let depMoveStart = eobtS + depRotS;
      if (depMoveStart > etotS) depMoveStart = eobtS;
      return { t0: depMoveStart, t1: etotS };
    }
    const eldtMin = flightEMinutesPrefer(f, ['eldtMin'], flightEMinutesPrefer(f, ['timeMin'], 0));
    const eibtMin = flightEMinutesPrefer(f, ['eibtMin'], eldtMin + 15);
    const eobtMin = flightEMinutesPrefer(f, ['eobtMin'], eibtMin + (typeof f.dwellMin === 'number' && isFinite(f.dwellMin) ? f.dwellMin : 45));
    const etotMin = flightEMinutesPrefer(f, ['etotMin'], eobtMin + 30);
    const eldtS = eldtMin * 60;
    const etotS = etotMin * 60;
    const tAppr = arrivalApproachDurationSecBeforeEldt(f);
    if (!isFinite(tAppr) || tAppr < 0) return null;
    const t0 = eldtS - tAppr;
    if (!isFinite(t0) || !isFinite(etotS)) return null;
    return { t0: t0, t1: etotS };
  }
  
  function simAirsideLazyPadSec() {
    return Math.max(90, SIM_TIME_SLIDER_SNAP_SEC + 45);
  }
  function isFlightAirsideActiveAtSimSec(f, tSec) {
    const w = getFlightAirsideWindowSec(f);
    if (!w || !isFinite(Number(tSec))) return false;
    const t = Number(tSec);
    return t >= w.t0 - 1e-3 && t <= w.t1 + 1e-3;
  }
  function isFlightAirsideLazyTimelineBuildEligible(f, tSec) {
    const w = getFlightAirsideWindowSec(f);
    if (!w || !isFinite(Number(tSec))) return false;
    const t = Number(tSec);
    const pad = simAirsideLazyPadSec();
    return t >= w.t0 - pad - 1e-3 && t <= w.t1 + 1e-3;
  }
  function nearestIndexOnPolylineForTd(pts, q) {
    if (!pts || pts.length < 2) return 0;
    let bestI = 0, bestD2 = Infinity;
    for (let i = 0; i < pts.length - 1; i++) {
      const pr = projectOnSegment(pts[i], pts[i + 1], q);
      const d2 = dist2(pr.p, q);
      if (d2 < bestD2) { bestD2 = d2; bestI = i; }
    }
    return bestI;
  }
  function trimPolylineFromNearPoint(pts, nearPt) {
    if (!pts || pts.length < 2) return pts ? pts.slice() : [];
    const idx = nearestIndexOnPolylineForTd(pts, nearPt);
    const a = pts[idx], b = pts[idx + 1];
    const pr = projectOnSegment(a, b, nearPt);
    const t = Math.max(0, Math.min(1, pr.t));
    const start = [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])];
    const out = [start];
    for (let j = idx + 1; j < pts.length; j++) out.push([pts[j][0], pts[j][1]]);
    return out.length >= 2 ? out : pts.slice();
  }
  function approachPointBeforeThresholdJs(runwayId, rwDir, offsetWorld, touchdownDistAlong) {
    const r = getRunwayPath(runwayId);
    const td = getRunwayPointAtDistance(runwayId, touchdownDistAlong);
    if (!r || !r.pts || r.pts.length < 2) return td || [0, 0];
    const pts = r.pts;
    let segIdx = Math.max(0, pts.length - 2);
    let acc = 0;
    for (let i = 0; i < pts.length - 1; i++) {
      const segLen = pathDist(pts[i], pts[i + 1]);
      if (segLen < 1e-9) continue;
      if (acc + segLen >= touchdownDistAlong - 1e-6) { segIdx = i; break; }
      acc += segLen;
    }
    const p1 = pts[segIdx], p2 = pts[segIdx + 1];
    const segLen = pathDist(p1, p2) || 1;
    let ux = (p2[0] - p1[0]) / segLen, uy = (p2[1] - p1[1]) / segLen;
    if (rwDir === 'counter_clockwise') { ux = -ux; uy = -uy; }
    return [td[0] - ux * offsetWorld, td[1] - uy * offsetWorld];
  }
  function mergeTimelineSegments(a, b) {
    if (!a || !a.length) return b ? b.slice() : [];
    if (!b || !b.length) return a.slice();
    const out = a.slice();
    const last = out[out.length - 1], first = b[0];
    if (Math.abs(last.t - first.t) < 1e-3 && Math.abs(last.x - first.x) < 0.1) out.pop();
    for (let i = 0; i < b.length; i++) out.push(b[i]);
    return out;
  }
  function polylineTotalLength(pts) {
    if (!pts || pts.length < 2) return 0;
    let s = 0;
    for (let i = 0; i < pts.length - 1; i++) s += pathDist(pts[i], pts[i + 1]);
    return s;
  }
  function polylinePointAtDistance(pts, distAlong) {
    if (!pts || !pts.length) return [0, 0];
    const d = Math.max(0, Number(distAlong) || 0);
    if (d <= 1e-12) return [pts[0][0], pts[0][1]];
    let acc = 0;
    for (let i = 0; i < pts.length - 1; i++) {
      const a = pts[i], b = pts[i + 1];
      const seg = pathDist(a, b);
      if (seg < 1e-9) continue;
      if (acc + seg >= d - 1e-9) {
        const t = Math.max(0, Math.min(1, (d - acc) / seg));
        return [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])];
      }
      acc += seg;
    }
    const last = pts[pts.length - 1];
    return [last[0], last[1]];
  }
  function polylineSplitAtDistance(pts, cutDist) {
    if (!pts || pts.length < 2) return { first: pts ? pts.slice() : [], second: [] };
    const cut = Math.max(0, Number(cutDist) || 0);
    if (cut <= 1e-9) return { first: [[pts[0][0], pts[0][1]]], second: pts.slice() };
    let acc = 0;
    const first = [[pts[0][0], pts[0][1]]];
    for (let i = 0; i < pts.length - 1; i++) {
      const a = pts[i], b = pts[i + 1];
      const seg = pathDist(a, b);
      if (seg < 1e-9) continue;
      if (acc + seg >= cut - 1e-9) {
        const t = Math.max(0, Math.min(1, (cut - acc) / seg));
        const px = a[0] + t * (b[0] - a[0]), py = a[1] + t * (b[1] - a[1]);
        if (dist2(first[first.length - 1], [px, py]) > 1e-8) first.push([px, py]);
        const second = [[px, py]];
        for (let j = i + 1; j < pts.length; j++) second.push([pts[j][0], pts[j][1]]);
        return { first: dedupePathPoints(first), second: dedupePathPoints(second) };
      }
      acc += seg;
      if (dist2(first[first.length - 1], b) > 1e-8) first.push([b[0], b[1]]);
    }
    return { first: dedupePathPoints(first), second: [[pts[pts.length - 1][0], pts[pts.length - 1][1]]] };
  }
  function aircraftDecelMs2ForTimeline(f) {
    const ac = (typeof getAircraftInfoByType === 'function') ? getAircraftInfoByType(f && f.aircraftType) : null;
    const a = ac && typeof ac.deceleration_avg_ms2 === 'number' ? ac.deceleration_avg_ms2 : null;
    if (typeof a === 'number' && isFinite(a) && a > 0.05) return Math.min(5, Math.max(0.05, a));
    return 1.2;
  }
  function nearestTaxiInfraD2ForMidpoint(mid) {
    let bestApronD2 = Infinity;
    let bestTaxiD2 = Infinity;
    let bestTw = null;
    const apronList = state.apronLinks || [];
    for (let ai = 0; ai < apronList.length; ai++) {
      const poly = getApronLinkPolylineWorldPts(apronList[ai]);
      if (!poly || poly.length < 2) continue;
      for (let j = 0; j < poly.length - 1; j++) {
        const pr = projectOnSegment(poly[j], poly[j + 1], mid);
        const d2 = dist2(pr.p, mid);
        if (d2 < bestApronD2) bestApronD2 = d2;
      }
    }
    const list = state.taxiways || [];
    for (let ti = 0; ti < list.length; ti++) {
      const tw = list[ti];
      const ot = getOrderedPoints(tw);
      if (!ot || ot.length < 2) continue;
      for (let j = 0; j < ot.length - 1; j++) {
        const pr = projectOnSegment(ot[j], ot[j + 1], mid);
        const d2 = dist2(pr.p, mid);
        if (d2 < bestTaxiD2) { bestTaxiD2 = d2; bestTw = tw; }
      }
    }
    return { bestApronD2, bestTaxiD2, bestTw };
  }
  function taxiHitFromMidpoint(mid) {
    const { bestApronD2, bestTaxiD2, bestTw } = nearestTaxiInfraD2ForMidpoint(mid);
    const hasA = bestApronD2 < Infinity;
    const hasT = bestTaxiD2 < Infinity;
    if (hasA && (!hasT || bestApronD2 <= bestTaxiD2)) return { kind: 'apron' };
    if (hasT && bestTw) return { kind: 'tw', tw: bestTw };
    return { kind: 'tw', tw: null };
  }
  function taxiSegmentVelocityMsFromHit(hit, carry) {
    const fallback = getTaxiwayAvgMoveVelocityForPath(null);
    if (hit.kind === 'apron') return Math.max(0.1, APRON_TAXIWAY_SPEED_MS);
    const tw = hit.tw;
    if (!tw) return Math.max(1, fallback);
    const pt = tw.pathType || 'taxiway';
    if (pt === 'runway_exit') {
      const v = carry.lastTaxiwayMs;
      return Math.max(1, (typeof v === 'number' && v > 0) ? v : fallback);
    }
    if (pt === 'taxiway') {
      const v = getTaxiwayAvgMoveVelocityForPath(tw);
      carry.lastTaxiwayMs = v;
      return Math.max(1, v);
    }
    if (pt === 'runway') return Math.max(1, getTaxiwayAvgMoveVelocityForPath(tw));
    return Math.max(1, getTaxiwayAvgMoveVelocityForPath(tw));
