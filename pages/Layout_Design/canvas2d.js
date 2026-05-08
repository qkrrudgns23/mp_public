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
        const standWindowOverlapInvalid = !!(segStandId && typeof flightWouldOverlapStandAssignment === 'function' && flightWouldOverlapStandAssignment(f, segStandId, segIdx));
        const standOverlapClass = standWindowOverlapInvalid ? ' alloc-stand-window-overlap' : '';
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
        const trDead = (typeof compactPlaybackTrackForFlight === 'function') ? compactPlaybackTrackForFlight(f) : null;
        const deadlockOverlayHtml = allocFlightDeadlockOverlayHtml(trDead, t0, t1, tStart, tEnd);
        const deadlockBarClass = allocFlightTrackHasDeadlock(trDead) ? ' alloc-flight-deadlock' : '';
        return '' +
          '<div class="alloc-flight' + conflictClass + invalidClass + standOverlapClass + selectedClass + sbarDimClass + deadlockBarClass + '" draggable="true" data-flight-id="' + f.id + '" data-segment-idx="' + segIdx + '" ' +
            'style="left:' + leftPct + '%;width:' + widthPct + '%;min-width:4px;"' +
            ' title="' + barTitle + '">' +
            deadlockOverlayHtml +
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
    const simMinPlay = state.simTimeSec / 60;
    const simPhVisible = isFinite(simMinPlay) && simMinPlay >= winStart - 1e-9 && simMinPlay <= winEnd + 1e-9;
    const simPhLeft = allocLeftPct(simMinPlay);
    const simPlayheadHtml =
      '<div class="alloc-gantt-sim-playhead-layer">' +
        '<div class="alloc-gantt-sim-playhead" style="left:' + simPhLeft + '%;' + (simPhVisible ? '' : 'display:none;') +
        '" title="Simulation time — drag to scrub"></div>' +
      '</div>';
    const trackColHtml =
      '<div class="alloc-gantt-scroll-col">' +
        '<div class="alloc-gantt-inner" style="min-width:' + innerMinWidthPct + '%;">' +
          gridOverlayHtml +
          simPlayheadHtml +
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
    if (!ganttEl._allocSimPlayheadPtrBound) {
      ganttEl._allocSimPlayheadPtrBound = true;
      ganttEl.addEventListener('pointerdown', function(ev) {
        const t = ev.target && ev.target.closest ? ev.target.closest('.alloc-gantt-sim-playhead') : null;
        if (!t || !ganttEl.contains(t)) return;
        if (ev.button != null && ev.button !== 0) return;
        if (ev.isPrimary === false) return;
        ev.preventDefault();
        ev.stopPropagation();
        state._allocGanttSimPlayheadDragging = true;
        try { t.setPointerCapture(ev.pointerId); } catch (e2) {}
      });
    }
    if (!window._allocGanttSimPlayheadDocWired) {
      window._allocGanttSimPlayheadDocWired = true;
      document.addEventListener('pointermove', function(ev) {
        if (!state._allocGanttSimPlayheadDragging) return;
        const gE = document.getElementById('allocationGantt');
        const m = _ganttClientXToMinutes(ev.clientX, gE);
        if (m == null || !isFinite(m)) return;
        const secRaw = m * 60;
        const lo = state.simStartSec, hi = state.simDurationSec;
        const sec = snapSimTimeSecForSlider(Math.max(lo, Math.min(hi, secRaw)));
        state.simTimeSec = sec;
        const sldr = document.getElementById('flightSimSlider');
        if (sldr) sldr.value = String(sec);
        if (typeof updateFlightSimPlaybackLabelsDom === 'function') updateFlightSimPlaybackLabelsDom();
        syncAllocGanttSimPlayheadPosition();
        try { draw(); } catch (e3) {}
        update3DSceneWhenVisible();
        ev.preventDefault();
      }, true);
      document.addEventListener('pointerup', function() {
        if (!state._allocGanttSimPlayheadDragging) return;
        state._allocGanttSimPlayheadDragging = false;
        try { draw(); } catch (e4) {}
        update3DSceneWhenVisible();
      }, true);
      document.addEventListener('pointercancel', function() {
        state._allocGanttSimPlayheadDragging = false;
      }, true);
    }
    syncAllocGanttSimPlayheadPosition();
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
  var _allocGanttPreviewLastKey = '';
  function _allocGanttDragStandPreviewAllowed(f, standId) {
    if (!standId) return true;
    var allStands = allStandsForFlightAssignment();
    var stand = allStands.find(function(s) { return s.id === standId; });
    if (!stand) return false;
    var ctx = state && state._allocGanttDrag;
    var segs = typeof normalizeFlightApronStaySegments === 'function' ? normalizeFlightApronStaySegments(f) : [];
    var segCount = Math.max(1, segs.length || 1);
    var segIdx = ctx && ctx.flightId === f.id && ctx.segmentIdx != null ? ctx.segmentIdx : 0;
    if (typeof flightCanUseStandForSegment === 'function' && !flightCanUseStandForSegment(f, stand, segIdx, segCount)) return false;
    if (typeof flightWouldOverlapStandAssignment === 'function' && flightWouldOverlapStandAssignment(f, standId, segIdx)) return false;
    return true;
  }
  /** Undo alloc Gantt drag preview (mutated stand/segments) when drop is rejected or cancelled. */
  function _allocGanttRevertUncommittedDragPreview(st) {
    var ctx = st._allocGanttDrag;
    if (!ctx || !ctx.flightId) return;
    var f = st.flights.find(function(x) { return x.id === ctx.flightId; });
    var restoredFromSegSnap = false;
    if (f && ctx.prevApronSegmentsJson) {
      try {
        var prevSegs = JSON.parse(ctx.prevApronSegmentsJson);
        if (Array.isArray(prevSegs) && prevSegs.length > 0) {
          f.apronStaySegments = prevSegs;
          if (typeof syncFlightApronStayAggregate === 'function') syncFlightApronStayAggregate(f);
          restoredFromSegSnap = true;
        }
      } catch (eRestore) {}
    }
    if (f && !restoredFromSegSnap) {
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
      var key = ctx.flightId + '|' + (ctx.segmentIdx != null ? ctx.segmentIdx : '') + '|' + (sid || '');
      if (key === _allocGanttPreviewLastKey) return;
      _allocGanttPreviewLastKey = key;
      if (ctx.segmentIdx != null && typeof normalizeFlightApronStaySegments === 'function') {
        var segs = normalizeFlightApronStaySegments(f);
        if (segs[ctx.segmentIdx]) {
          segs[ctx.segmentIdx].standId = sid;
          f.apronStaySegments = segs;
          if (typeof syncFlightApronStayAggregate === 'function') syncFlightApronStayAggregate(f);
        }
      } else {
        f.standId = sid;
        if (f.token) f.token.apronId = sid;
      }
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
      _allocGanttRevertUncommittedDragPreview(st);
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
        if (track.getAttribute('data-apron-link-ok') === '0') {
          showAllocationConstraintModal("This stand has no Leadin Taxiway link, so it cannot accept a flight.");
          _allocGanttRevertUncommittedDragPreview(st);
          return;
        }
        const flightId = ev.dataTransfer.getData('text/plain');
        if (!flightId) return;
        const f = st.flights.find(function(x) { return x.id === flightId; });
        if (!f) return;
        const segIdx = st._allocGanttDrag && st._allocGanttDrag.flightId === flightId ? st._allocGanttDrag.segmentIdx : null;
        if (!assignStandToFlight(f, track.getAttribute('data-stand-id') || null, segIdx, { fromAllocGantt: true })) {
          _allocGanttRevertUncommittedDragPreview(st);
          return;
        }
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
    if (!ganttEl._allocSplitBound) {
      ganttEl._allocSplitBound = true;
      function applySplitAtEvent(ev, btn) {
        const flightEl = btn ? btn.closest('.alloc-flight') : (ev.target && ev.target.closest ? ev.target.closest('.alloc-flight') : null);
        if (!flightEl) return;
        const flightId = flightEl.getAttribute('data-flight-id');
        const f = st.flights.find(function(x) { return String(x.id) === String(flightId); });
        if (!f) return;
        const segIdx = parseInt((btn && btn.getAttribute('data-segment-idx')) || flightEl.getAttribute('data-segment-idx') || '0', 10) || 0;
        let rawM = null;
        if (btn) {
          const segs = typeof normalizeFlightApronStaySegments === 'function' ? normalizeFlightApronStaySegments(f) : [];
          const seg = segs[segIdx];
          if (seg && isFinite(seg.sibtMin) && isFinite(seg.sobtMin)) rawM = (seg.sibtMin + seg.sobtMin) * 0.5;
        } else {
          rawM = _ganttClientXToMinutes(ev.clientX, ganttEl);
        }
        if (rawM == null || !isFinite(rawM)) return;
        if (!splitFlightApronStaySegmentAtMinute(f, segIdx, rawM)) return;
        if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
        const touched = [];
        (f.apronStaySegments || []).forEach(function(seg) {
          if (seg && seg.standId) touched.push(seg.standId);
        });
        if (typeof renderFlightList === 'function') {
          renderFlightList(false, false, { scheduleMode: 'incremental', dirtyFlightIds: [f.id], touchedStandIds: touched, skipGanttRefresh: true });
        }
        if (typeof renderFlightGantt === 'function') renderFlightGantt({ skipPathPrep: true });
      }
      ganttEl.addEventListener('click', function(ev) {
        const btn = ev.target && ev.target.closest ? ev.target.closest('.alloc-flight-split-btn') : null;
        if (!btn) return;
        ev.preventDefault();
        ev.stopPropagation();
        applySplitAtEvent(ev, btn);
      }, true);
      ganttEl.addEventListener('click', function(ev) {
        if (!ev.altKey) return;
        const flightEl = ev.target && ev.target.closest ? ev.target.closest('.alloc-flight') : null;
        if (!flightEl || ev.target.closest('.alloc-flight-handle')) return;
        ev.preventDefault();
        ev.stopPropagation();
        applySplitAtEvent(ev, null);
      }, true);
    }
    ganttEl.querySelectorAll('.alloc-flight').forEach(function(el) {
      el.addEventListener('dragstart', function(ev) {
        if (ev.target && ev.target.closest && (ev.target.closest('.alloc-flight-handle') || ev.target.closest('.alloc-flight-split-btn'))) {
          ev.preventDefault();
          return;
        }
        var flightId = this.getAttribute('data-flight-id') || '';
        var segIdx = parseInt(this.getAttribute('data-segment-idx') || '0', 10) || 0;
        ev.dataTransfer.setData('text/plain', flightId);
        ev.dataTransfer.effectAllowed = 'move';
        var fDrag = st.flights.find(function(x) { return x.id === flightId; });
        if (fDrag) {
          var prevSegsJson = '';
          if (typeof normalizeFlightApronStaySegments === 'function') {
            prevSegsJson = JSON.stringify(normalizeFlightApronStaySegments(fDrag).map(function(seg) {
              return { standId: seg.standId || null, sibtMin: seg.sibtMin, sobtMin: seg.sobtMin };
            }));
          }
          st._allocGanttDragSeq = (st._allocGanttDragSeq || 0) + 1;
          st._allocGanttDrag = {
            flightId: flightId,
            segmentIdx: segIdx,
            prevStandId: fDrag.standId || null,
            prevApron: (fDrag.token && fDrag.token.apronId) ? fDrag.token.apronId : null,
            prevApronSegmentsJson: prevSegsJson,
            seq: st._allocGanttDragSeq
          };
          st._allocGanttDropHandled = false;
          _allocGanttPreviewLastKey = '';
        }
      });
      el.addEventListener('click', function(ev) {
        if (ev.target && ev.target.closest && ev.target.closest('.alloc-flight-handle')) return;
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
        if (ev.target && ev.target.closest && ev.target.closest('.alloc-flight-handle')) return;
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
        if (this.getAttribute('data-apron-link-ok') === '0') {
          showAllocationConstraintModal("This stand has no Leadin Taxiway link, so it cannot accept a flight.");
          _allocGanttRevertUncommittedDragPreview(st);
          return;
        }
        const flightId = ev.dataTransfer.getData('text/plain');
        if (!flightId) return;
        const f = st.flights.find(function(x) { return x.id === flightId; });
        if (!f) return;
        const segIdx = st._allocGanttDrag && st._allocGanttDrag.flightId === flightId ? st._allocGanttDrag.segmentIdx : null;
        if (!assignStandToFlight(f, this.getAttribute('data-stand-id') || null, segIdx, { fromAllocGantt: true })) {
          _allocGanttRevertUncommittedDragPreview(st);
          return;
        }
        st._allocGanttDropHandled = true;
      });
    });
    if (!ganttEl._ganttSibtSobtPointerBound) {
      ganttEl._ganttSibtSobtPointerBound = true;
      ganttEl.addEventListener('pointerdown', function(ev) {
        const h = ev.target && ev.target.closest ? ev.target.closest('.alloc-flight-handle') : null;
        if (!h) return;
        const flightEl = h.closest('.alloc-flight');
        if (!flightEl) return;
        const fid = flightEl.getAttribute('data-flight-id');
        if (!fid) return;
        const f = st.flights.find(function(x) { return String(x.id) === String(fid); });
        if (!f || (typeof flightBlockedLikeNoWay === 'function' && flightBlockedLikeNoWay(f))) return;
        const role = h.getAttribute('data-handle-role');
        if (role !== 'sibt' && role !== 'sobt') return;
        const handleSegIdx = parseInt(h.getAttribute('data-segment-idx') || flightEl.getAttribute('data-segment-idx') || '0', 10) || 0;
        ev.preventDefault();
        ev.stopPropagation();
        st._allocGanttHandleDragViewportPin = {
          active: true,
          winStart0: (st.allocGanttWindowStartMin != null && isFinite(st.allocGanttWindowStartMin)) ? st.allocGanttWindowStartMin : null
        };
        const tip = _ganttEnsureSibtSobtTooltipEl();
        const pid = ev.pointerId;
        const b0 = (typeof getNormalizedStandDwellBounds === 'function') ? getNormalizedStandDwellBounds(f) : { dwell: 0, minDwell: 0 };
        const startSibt0 = (f.sibtMin != null && isFinite(f.sibtMin)) ? f.sibtMin : (f.timeMin != null ? f.timeMin : 0);
        let anchorSobt0 = (f.sobtMin != null && isFinite(f.sobtMin)) ? f.sobtMin : (startSibt0 + b0.dwell);
        const dragSibtCtx = {
          anchorSobt: anchorSobt0,
          startSibt: startSibt0,
          startEibt: f.eibtMin,
          startEobt: f.eobtMin,
          startEtot: f.etotMin,
          dwell0: b0.dwell,
          minDwell0: b0.minDwell
        };
        let rafPending = null;
        function flushUi() {
          rafPending = null;
          const touched = [];
          if (f.standId) touched.push(f.standId);
          if (typeof renderFlightList === 'function') {
            renderFlightList(false, false, { scheduleMode: 'incremental', dirtyFlightIds: [f.id], touchedStandIds: touched, skipGanttRefresh: true });
          }
          if (typeof renderFlightGantt === 'function') renderFlightGantt({ skipPathPrep: true });
        }
        function applyAtClientX(cx, tipX, tipY) {
          const rawM = _ganttClientXToMinutes(cx, ganttEl);
          if (rawM == null || !isFinite(rawM)) return;
          const snap = GANTT_SIBT_SOBT_HANDLE_SNAP_MIN;
          let m = Math.max(0, Math.round(rawM / snap) * snap);
          const hasMultiSegments = Array.isArray(f.apronStaySegments) && f.apronStaySegments.length > 1;
          if (hasMultiSegments && typeof applyApronStaySegmentHandleMinute === 'function') {
            applyApronStaySegmentHandleMinute(f, handleSegIdx, role, m);
          } else if (role === 'sibt') {
            if (typeof _ganttApplySibtHandleSnappedMinutes === 'function') {
              _ganttApplySibtHandleSnappedMinutes(f, m, dragSibtCtx);
              if (typeof syncSingleApronStaySegmentFromAggregate === 'function') syncSingleApronStaySegmentFromAggregate(f);
            }
          } else {
            if (typeof applyScheduledGateTimingFromSField === 'function') {
              applyScheduledGateTimingFromSField(f, 'sobt', m);
              if (typeof syncSingleApronStaySegmentFromAggregate === 'function') syncSingleApronStaySegmentFromAggregate(f);
            }
          }
          if (typeof computeScheduledDisplayTimesIncremental === 'function') {
            const tset = new Set();
            if (f.standId) tset.add(f.standId);
            computeScheduledDisplayTimesIncremental(st.flights, new Set([f.id]), tset);
          }
          if (role === 'sibt') {
            const showVal = hasMultiSegments && f.apronStaySegments && f.apronStaySegments[handleSegIdx]
              ? f.apronStaySegments[handleSegIdx].sibtMin
              : f.timeMin;
            tip.textContent = (hasMultiSegments ? ('SIBT' + (handleSegIdx + 1)) : 'SIBT') + ': ' + formatFlightScheduleDateTime(f, showVal);
          } else {
            const sobtShow = hasMultiSegments && f.apronStaySegments && f.apronStaySegments[handleSegIdx]
              ? f.apronStaySegments[handleSegIdx].sobtMin
              : (f.sobtMin != null ? f.sobtMin : m);
            tip.textContent = (hasMultiSegments ? ('SOBT' + (handleSegIdx + 1)) : 'SOBT') + ': ' + formatFlightScheduleDateTime(f, sobtShow);
          }
          tip.removeAttribute('hidden');
          tip.style.left = Math.min(window.innerWidth - 200, Math.max(8, tipX + 12)) + 'px';
          tip.style.top = Math.min(window.innerHeight - 40, Math.max(8, tipY + 12)) + 'px';
          if (rafPending == null) rafPending = requestAnimationFrame(flushUi);
        }
        applyAtClientX(ev.clientX, ev.clientX, ev.clientY);
        function onMove(e) {
          if (e.pointerId !== pid) return;
          applyAtClientX(e.clientX, e.clientX, e.clientY);
        }
        function onUp() {
          document.removeEventListener('pointermove', onMove);
          document.removeEventListener('pointerup', onUp);
          document.removeEventListener('pointercancel', onUp);
          st._allocGanttHandleDragViewportPin = null;
          if (rafPending != null) {
            cancelAnimationFrame(rafPending);
            rafPending = null;
          }
          flushUi();
          tip.setAttribute('hidden', '');
        }
        document.addEventListener('pointermove', onMove);
        document.addEventListener('pointerup', onUp);
        document.addEventListener('pointercancel', onUp);
      }, true);
    }
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

  function kpiNormalizeScheduleDateForKpi(raw) {
    const s = (raw == null ? '' : String(raw)).trim();
    if (/^\d{4}-\d{2}-\d{2}$/.test(s)) return s;
    return typeof DEFAULT_SIBT_DATE !== 'undefined' ? DEFAULT_SIBT_DATE : '1970-01-01';
  }
  function kpiScheduleDayStartAbsMin(dateStr) {
    const parts = String(dateStr).split('-');
    if (parts.length !== 3) return 0;
    const y = parseInt(parts[0], 10), mo = parseInt(parts[1], 10) - 1, d = parseInt(parts[2], 10);
    if (!isFinite(y) || !isFinite(mo) || !isFinite(d)) return 0;
    return Math.floor(Date.UTC(y, mo, d) / 60000);
  }
  /** Minutes since Unix epoch (UTC midnight of schedule date + minute-of-day). */
  function kpiFlightScheduleAbsMinute(f, minuteFromMidnight) {
    const m = kpiToNumber(minuteFromMidnight);
    if (m == null || !isFinite(m)) return null;
    const day0 = kpiScheduleDayStartAbsMin(kpiNormalizeScheduleDateForKpi(f && (f.sibtDate != null ? f.sibtDate : f.serviceDate)));
    const mod = ((Math.floor(m) % 1440) + 1440) % 1440;
    return day0 + mod;
  }
  function kpiApronStandCountState() {
    const p = (state.pbbStands || []).length;
    const r = (state.remoteStands || []).length;
    const t = (state.tempStands || []).length;
    const n = p + r + t;
    return Math.max(1, n);
  }
  function kpiFormatOptionalSecondsAvg(value) {
    const v = kpiToNumber(value);
    if (v == null || !isFinite(v)) return '—';
    return kpiFormatSecondsValue(v);
  }
  function kpiFormatOptionalMinutesAvg(value) {
    const v = kpiToNumber(value);
    if (v == null || !isFinite(v)) return '—';
    return kpiFormatMinutesValue(v);
  }
  function kpiFormatRatioPercent(value) {
    const v = kpiToNumber(value);
    if (v == null || !isFinite(v)) return '—';
    return (v * 100).toFixed(1) + '%';
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

  function kpiDisposeInteractiveCharts() {
    try {
      if (window.__kpiChartRunway) { window.__kpiChartRunway.destroy(); window.__kpiChartRunway = null; }
    } catch (e) { console.warn('kpiDisposeInteractiveCharts', e); }
  }
  function kpiRunwayHourlyChartOptions(labelsLen) {
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
          padding: 10
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255,255,255,0.07)' },
          ticks: {
            color: '#94a3b8',
            maxRotation: labelsLen > 18 ? 45 : 0,
            autoSkip: labelsLen > 24,
            maxTicksLimit: labelsLen > 32 ? 18 : undefined,
            font: { size: 11 }
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
  function kpiMountRunwayHourlyChart(series) {
    if (typeof Chart === 'undefined') {
      console.warn('Chart.js failed to load; KPI charts are static until CDN is available.');
      return;
    }
    if (!series || !series.labels || !series.labels.length) return;
    const elR = document.getElementById('kpiChartRunway');
    if (!elR) return;
    const opt = kpiRunwayHourlyChartOptions(series.labels.length);
    window.__kpiChartRunway = new Chart(elR, {
      type: 'line',
      data: {
        labels: series.labels,
        datasets: [
          {
            label: 'Total',
            data: series.total,
            borderColor: '#c4b5fd',
            backgroundColor: 'transparent',
            borderWidth: 2.5,
            tension: 0.2,
            pointRadius: 3,
            pointHoverRadius: 6,
            order: 1
          },
          {
            label: 'Arrivals',
            data: series.arr,
            borderColor: '#38bdf8',
            backgroundColor: 'transparent',
            borderWidth: 2,
            tension: 0.2,
            pointRadius: 3,
            pointHoverRadius: 6,
            order: 2
          },
          {
            label: 'Departures',
            data: series.dep,
            borderColor: '#fb923c',
            backgroundColor: 'transparent',
            borderWidth: 2,
            tension: 0.2,
            pointRadius: 3,
            pointHoverRadius: 6,
            order: 3
          }
        ]
      },
      options: opt
    });
  }
  function kpiRunwayChartPlaceholder(hasHourly) {
    if (!hasHourly) return '<div class="kpi-empty-state">No runway movements in the flight schedule for the hourly chart.</div>';
    return '<div class="kpi-chart-canvas-host"><canvas id="kpiChartRunway" aria-label="Hourly runway traffic chart"></canvas></div>';
  }

  function collectKpiSnapshot() {
    const flights = Array.isArray(state.flights) ? state.flights.slice() : [];
    const rows = flights.map(function(f) {
      const isDepOnly = !!(f && f.arrDep === 'Dep');
      const arrTaxiBase = kpiToNumber(typeof getBaseVttArrMinutes === 'function' ? getBaseVttArrMinutes(f) : null);
      const depBlockOutMin = kpiToNumber(typeof getDepBlockOutMin === 'function' ? getDepBlockOutMin(f) : null);
      const depTaxiBase = kpiToNumber(typeof getBaseVttDepMinutesToLineup === 'function' ? getBaseVttDepMinutesToLineup(f) : null);
      const arrTaxiFromSchedSec = (f && f.proSimVttArrSec != null && isFinite(Number(f.proSimVttArrSec))) ? Number(f.proSimVttArrSec) / 60 : null;
      const depTaxiFromSchedSec = (f && f.proSimVttDepSec != null && isFinite(Number(f.proSimVttDepSec))) ? Number(f.proSimVttDepSec) / 60 : null;
      const arrTaxiSch = kpiToNumber(arrTaxiFromSchedSec != null ? arrTaxiFromSchedSec : arrTaxiBase);
      const depTaxiSch = kpiToNumber(depTaxiFromSchedSec != null ? depTaxiFromSchedSec : depTaxiBase);
      const rotSec = kpiToNumber(f && f.arrRotSec != null ? f.arrRotSec : (typeof getArrRotMinutes === 'function' ? getArrRotMinutes(f) * 60 : null));
      const depRotFromSchedSec = (f && f.proSimDepLineupSec != null && isFinite(Number(f.proSimDepLineupSec))) ? Number(f.proSimDepLineupSec) : null;
      const depRotSec = kpiToNumber(depRotFromSchedSec != null ? depRotFromSchedSec : (
        (typeof SCHED_DEP_ROT_MIN === 'number' && isFinite(SCHED_DEP_ROT_MIN)) ? SCHED_DEP_ROT_MIN * 60 : null
      ));
      const sibt = kpiToNumber(f && f.sibtMin != null ? f.sibtMin : (f && f.timeMin != null ? f.timeMin : null));
      const sldt = kpiToNumber(f && f.sldtMin != null ? f.sldtMin : (sibt != null && arrTaxiSch != null && rotSec != null ? Math.max(0, sibt - arrTaxiSch - rotSec / 60) : null));
      const dwellMin = kpiToNumber(f && f.dwellMin != null ? f.dwellMin : null);
      const sobt = kpiToNumber(f && f.sobtMin != null ? f.sobtMin : (sibt != null && dwellMin != null ? sibt + dwellMin : null));
      const sttDepMinK = kpiToNumber(typeof getBaseVttDepMinutesToHoldingSlot === 'function' ? getBaseVttDepMinutesToHoldingSlot(f) : depTaxiSch);
      const depRotMinK = depRotSec != null && isFinite(depRotSec) ? depRotSec / 60 : null;
      const stot = kpiToNumber(f && f.stotMin != null ? f.stotMin : (sobt != null && depRotMinK != null && sttDepMinK != null ? sobt + depRotMinK + sttDepMinK : (sobt != null && depBlockOutMin != null ? sobt + depBlockOutMin : null)));
      const eldtSched = kpiToNumber(sldt);
      const etotSched = kpiToNumber(stot);
      const failed = !!(f && flightBlockedLikeNoWay(f));
      return {
        flight: f,
        isDepOnly: isDepOnly,
        arrTaxiSch: arrTaxiSch,
        depTaxiSch: depTaxiSch,
        rotSec: rotSec,
        depRotSec: depRotSec,
        sibt: sibt,
        sobt: sobt,
        eldtSched: eldtSched,
        etotSched: etotSched,
        failed: failed
      };
    });
    const hourCounts = Object.create(null);
    function bumpRunwayHour(absMin, kind) {
