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
      var f = st.flights.find(function(x) { return x.id === ctx.flightId; });
      if (f && ctx.prevApronSegmentsJson) {
        try {
          var prevSegs = JSON.parse(ctx.prevApronSegmentsJson);
          if (Array.isArray(prevSegs)) {
            f.apronStaySegments = prevSegs;
            if (typeof syncFlightApronStayAggregate === 'function') syncFlightApronStayAggregate(f);
          }
        } catch (eRestore) {}
      } else if (f) {
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
        const segIdx = st._allocGanttDrag && st._allocGanttDrag.flightId === flightId ? st._allocGanttDrag.segmentIdx : null;
        if (!assignStandToFlight(f, track.getAttribute('data-stand-id') || null, segIdx)) return;
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
        if (this.getAttribute('data-apron-link-ok') === '0') return;
        const flightId = ev.dataTransfer.getData('text/plain');
        if (!flightId) return;
        const f = st.flights.find(function(x) { return x.id === flightId; });
        if (!f) return;
        const segIdx = st._allocGanttDrag && st._allocGanttDrag.flightId === flightId ? st._allocGanttDrag.segmentIdx : null;
        if (!assignStandToFlight(f, this.getAttribute('data-stand-id') || null, segIdx)) return;
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
            if (typeof _ganttApplySibtHandleSnappedMinutes === 'function') _ganttApplySibtHandleSnappedMinutes(f, m, dragSibtCtx);
          } else {
            if (typeof applyScheduledGateTimingFromSField === 'function') applyScheduledGateTimingFromSField(f, 'sobt', m);
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
      if (absMin == null || !isFinite(absMin)) return;
      const hk = Math.floor(absMin / 60);
      if (!hourCounts[hk]) hourCounts[hk] = { arr: 0, dep: 0 };
      hourCounts[hk][kind] += 1;
    }
    rows.forEach(function(r) {
      if (r.failed) return;
      if (!r.isDepOnly && r.eldtSched != null) {
        const am = kpiFlightScheduleAbsMinute(r.flight, r.eldtSched);
        bumpRunwayHour(am, 'arr');
      }
      if (r.isDepOnly) {
        if (r.etotSched != null) {
          const am = kpiFlightScheduleAbsMinute(r.flight, r.etotSched);
          bumpRunwayHour(am, 'dep');
        }
      } else if (r.etotSched != null) {
        const am = kpiFlightScheduleAbsMinute(r.flight, r.etotSched);
        bumpRunwayHour(am, 'dep');
      }
    });
    const hourKeys = Object.keys(hourCounts).map(function(x) { return parseInt(x, 10); }).filter(function(x) { return isFinite(x); }).sort(function(a, b) { return a - b; });
    const hourlyRunway = hourKeys.map(function(hk) {
      const c = hourCounts[hk];
      const arr = c.arr || 0, dep = c.dep || 0;
      const absMin0 = hk * 60;
      const dayOff = Math.floor(absMin0 / 1440);
      const mod = ((absMin0 % 1440) + 1440) % 1440;
      const hh = Math.floor(mod / 60);
      const label = (dayOff > 0 ? ('D+' + String(dayOff) + ' ') : '') + String(hh).padStart(2, '0') + ':00';
      return { hourKey: hk, label: label, arrivals: arr, departures: dep, total: arr + dep };
    });
    const hourlyChart = {
      labels: hourlyRunway.map(function(h) { return h.label; }),
      arr: hourlyRunway.map(function(h) { return h.arrivals; }),
      dep: hourlyRunway.map(function(h) { return h.departures; }),
      total: hourlyRunway.map(function(h) { return h.total; })
    };
    let minAbs = Infinity, maxAbs = -Infinity;
    let utilMinTotal = 0;
    rows.forEach(function(r) {
      if (r.failed) return;
      const a0 = kpiFlightScheduleAbsMinute(r.flight, r.sibt);
      const a1 = kpiFlightScheduleAbsMinute(r.flight, r.sobt);
      if (a0 != null) { minAbs = Math.min(minAbs, a0); maxAbs = Math.max(maxAbs, a0); }
      if (a1 != null) { minAbs = Math.min(minAbs, a1); maxAbs = Math.max(maxAbs, a1); }
      if (a0 != null && a1 != null && a1 > a0) utilMinTotal += (a1 - a0);
    });
    const windowMin = (isFinite(minAbs) && isFinite(maxAbs) && maxAbs > minAbs) ? (maxAbs - minAbs) : 0;
    const standN = kpiApronStandCountState();
    const totalStandMin = standN * windowMin;
    const apronUtilRatio = (totalStandMin > 1e-6) ? (utilMinTotal / totalStandMin) : null;
    const okRows = rows.filter(function(r) { return !r.failed; });
    const arrLegRows = okRows.filter(function(r) { return !r.isDepOnly; });
    const depLegRows = okRows.filter(function(r) { return r.isDepOnly || r.etotSched != null; });
    const rotArrAvgSec = kpiAverage(arrLegRows, function(r) { return r.rotSec; });
    const arrTaxiAvgMin = kpiAverage(arrLegRows, function(r) { return r.arrTaxiSch; });
    const depTaxiAvgMin = kpiAverage(depLegRows, function(r) { return r.depTaxiSch; });
    const rotDepAvgSec = kpiAverage(depLegRows, function(r) { return r.depRotSec; });
    const failedFlights = rows.filter(function(r) { return r.failed; });
    return {
      rows: rows,
      totalFlights: rows.length,
      failedFlights: failedFlights.length,
      hourlyChart: hourlyChart,
      hasHourlyRunway: hourKeys.length > 0,
      apronStandCount: standN,
      apronWindowMin: windowMin,
      apronUtilMin: utilMinTotal,
      apronUtilRatio: apronUtilRatio,
      rotArrAvgSec: rotArrAvgSec,
      arrTaxiAvgMin: arrTaxiAvgMin,
      depTaxiAvgMin: depTaxiAvgMin,
      rotDepAvgSec: rotDepAvgSec
    };
  }

  function renderKpiDashboard(reasonLabel) {
    const host = document.getElementById('kpiDashboard');
    const status = document.getElementById('kpiSnapshotStatus');
    if (!host) return;
    kpiDisposeInteractiveCharts();
    const snapshot = collectKpiSnapshot();
    if (!snapshot.totalFlights) {
      host.innerHTML = '<div class="kpi-empty-state">Flight schedule에 항공편이 없습니다. 스케줄을 추가하거나 불러온 뒤 KPI를 확인하세요.</div>';
      if (status) status.textContent = (reasonLabel || 'Snapshot') + ' · Flight schedule · ' + kpiFormatSnapshotTime();
      return;
    }
    const apronMeta = 'Stands ' + kpiFormatCount(snapshot.apronStandCount) +
      ' · Window ' + (snapshot.apronWindowMin > 0 ? (snapshot.apronWindowMin / 60).toFixed(1) + ' h span' : '—') +
      ' · Utilization Σ(SOBT−SIBT) ' + kpiFormatMinutesValue(snapshot.apronUtilMin) + ' min (flight schedule gate block)';
    const summaryCards = [
      kpiBuildSummaryCard('Total flights', kpiFormatCount(snapshot.totalFlights), 'accent'),
      kpiBuildSummaryCard('Schedule block / path issues', kpiFormatCount(snapshot.failedFlights), snapshot.failedFlights > 0 ? 'danger' : 'success')
    ].join('');
    const panelHtml = [
      kpiBuildPanel('Flight schedule · movement (averages)', 'ROT(Arr/Dep) · VTT(Arr/Dep) from schedule', [
        kpiBuildMetricRow('Avg arrival ROT', kpiFormatOptionalSecondsAvg(snapshot.rotArrAvgSec), 'ROT(Arr) · flight schedule avg · sec'),
        kpiBuildMetricRow('Avg Arrival VTT', kpiFormatOptionalMinutesAvg(snapshot.arrTaxiAvgMin), 'VTT(Arr) · flight schedule avg · min'),
        kpiBuildMetricRow('Avg Departure VTT', kpiFormatOptionalMinutesAvg(snapshot.depTaxiAvgMin), 'Dep_taxi after pushback finished · VTT_DEP_SEC · min'),
        kpiBuildMetricRow('Avg Departure ROT', kpiFormatOptionalSecondsAvg(snapshot.rotDepAvgSec), 'DEP_ROT_SEC (ETOT - E_LINEUP) · else tier depRotMin · avg sec')
      ]),
      kpiBuildPanel('Apron utilization', 'ratio = utilization / (stands × window)', [
        kpiBuildMetricRow('Utilization ratio', kpiFormatRatioPercent(snapshot.apronUtilRatio), apronMeta)
      ])
    ].join('');
    host.innerHTML = '' +
      '<div class="kpi-summary-grid">' + summaryCards + '</div>' +
      '<div class="kpi-panel-grid">' + panelHtml + '</div>' +
      '<div class="kpi-chart-grid">' +
        '<div class="kpi-chart-card kpi-chart-card-primary kpi-chart-card--runway-only">' +
          '<div class="kpi-chart-head">' +
            '<div>' +
              '<div class="kpi-chart-title">Hourly runway traffic</div>' +
              '<div class="kpi-chart-subtitle">Flight schedule · 정각 시각(시간)별 SLDT 착륙·STOT 출발 건수 · line: total / arrivals / departures</div>' +
            '</div>' +
            '<div class="kpi-chart-legend">' +
              '<span class="kpi-legend-item"><span class="kpi-legend-swatch" style="background:#c4b5fd;"></span>Total</span>' +
              '<span class="kpi-legend-item"><span class="kpi-legend-swatch" style="background:#38bdf8;"></span>Arrivals</span>' +
              '<span class="kpi-legend-item"><span class="kpi-legend-swatch" style="background:#fb923c;"></span>Departures</span>' +
            '</div>' +
          '</div>' +
          kpiRunwayChartPlaceholder(snapshot.hasHourlyRunway) +
        '</div>' +
      '</div>';
    if (status) status.textContent = (reasonLabel || 'Snapshot') + ' · Flight schedule · ' + kpiFormatSnapshotTime();
    kpiMountRunwayHourlyChart(snapshot.hourlyChart);
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
      f.sldtMin = m;
      const sibt = Math.max(0, m + vttArrMin + rotArrMin);
      f.timeMin = sibt;
      f.sibtMin = sibt;
      f.sobtMin = sibt + dwell;
      f.stotMin = scheduledStotFromSobtMinutes(f, f.sobtMin);
      f.dwellMin = dwell;
      f.minDwellMin = minDwell;
      return true;
    }
    if (field === 'sibt') {
      f.timeMin = m;
      f.sibtMin = m;
      f.sldtMin = scheduledSldtFromSibtMinutes(f, m);
      f.sobtMin = m + dwell;
      f.stotMin = scheduledStotFromSobtMinutes(f, f.sobtMin);
      f.dwellMin = dwell;
      f.minDwellMin = minDwell;
      return true;
    }
    if (field === 'sobt') {
      const sibt = f.timeMin != null ? f.timeMin : 0;
      let sobtAdj = Math.max(m, sibt + minDwell);
      f.sobtMin = sobtAdj;
      f.dwellMin = Math.max(SCHED_DWELL_FLOOR_MIN, sobtAdj - sibt);
      if (f.minDwellMin != null) {
        f.minDwellMin = Math.max(SCHED_DWELL_FLOOR_MIN, Math.min(f.dwellMin, f.minDwellMin));
      }
      f.stotMin = scheduledStotFromSobtMinutes(f, f.sobtMin);
      return true;
    }
    if (field === 'stot') {
      const sibt = f.timeMin != null ? f.timeMin : 0;
      const sobtGuess = scheduledSobtFromStotMinutes(f, m);
      let sobtAdj = Math.max(sobtGuess, sibt + minDwell);
      f.sobtMin = sobtAdj;
      f.dwellMin = Math.max(SCHED_DWELL_FLOOR_MIN, sobtAdj - sibt);
      if (f.minDwellMin != null) {
        f.minDwellMin = Math.max(SCHED_DWELL_FLOOR_MIN, Math.min(f.dwellMin, f.minDwellMin));
      }
      f.stotMin = scheduledStotFromSobtMinutes(f, f.sobtMin);
      return true;
    }
    return false;
  }

  function applySOffsetsFromSibtSobt(f) {
    if (!f || flightBlockedLikeNoWay(f)) return;
    const sibt = f.sibtMin;
    const sobt = f.sobtMin;
    if (typeof sibt === 'number' && isFinite(sibt)) {
      f.sldtMin = Math.max(0, sibt - SCHED_SIBT_MINUS_SLDT_MIN);
    }
    if (typeof sobt === 'number' && isFinite(sobt)) {
      f.stotMin = sobt + SCHED_STOT_MINUS_SOBT_MIN;
    }
  }

  function computeScheduledDisplayTimes(flights) {
    if (!flights || !flights.length) return;
    flights.forEach(f => {
      if (flightBlockedLikeNoWay(f)) return;
      const tArrMin = f.sibtMin != null ? f.sibtMin : (f.timeMin != null ? f.timeMin : 0);
      let dwell = f.dwellMin != null ? f.dwellMin : 0;
      let minDwell = f.minDwellMin != null ? f.minDwellMin : 0;
      dwell = Math.max(SCHED_DWELL_FLOOR_MIN, dwell);
      minDwell = Math.max(SCHED_DWELL_FLOOR_MIN, minDwell);
      if (minDwell > dwell) minDwell = dwell;
      f.dwellMin = dwell;
      f.minDwellMin = minDwell;
      const sobt = f.sobtMin != null ? Math.max(f.sobtMin, tArrMin + minDwell) : (tArrMin + dwell);
      f.timeMin = tArrMin;
      f.sibtMin = tArrMin;
      f.sobtMin = sobt;
      f.dwellMin = Math.max(SCHED_DWELL_FLOOR_MIN, sobt - tArrMin);
      applySOffsetsFromSibtSobt(f);
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
      const tArrMin = f.sibtMin != null ? f.sibtMin : (f.timeMin != null ? f.timeMin : 0);
      let dwell = f.dwellMin != null ? f.dwellMin : 0;
      let minDwell = f.minDwellMin != null ? f.minDwellMin : 0;
      dwell = Math.max(SCHED_DWELL_FLOOR_MIN, dwell);
      minDwell = Math.max(SCHED_DWELL_FLOOR_MIN, minDwell);
      if (minDwell > dwell) minDwell = dwell;
      f.dwellMin = dwell;
      f.minDwellMin = minDwell;
      const sobt = f.sobtMin != null ? Math.max(f.sobtMin, tArrMin + minDwell) : (tArrMin + dwell);
      f.timeMin = tArrMin;
      f.sibtMin = tArrMin;
      f.sobtMin = sobt;
      f.dwellMin = Math.max(SCHED_DWELL_FLOOR_MIN, sobt - tArrMin);
      applySOffsetsFromSibtSobt(f);
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
        const rotM = (ev.rotArrMin != null && isFinite(ev.rotArrMin)) ? ev.rotArrMin : getArrRotMinutes(ev.flight);
        const eibtMin = (ev.flight.eldtMin != null ? ev.flight.eldtMin : 0) + rotM + (ev.vttArrMin || 0);
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
      const sldtMin = f.sldtMin != null ? f.sldtMin : 0;
      const stotMin = f.stotMin != null ? f.stotMin : 0;
      const sobtMin = f.sobtMin != null ? f.sobtMin : 0;
      const vttArrMin = getBaseVttArrMinutes(f);
      const rotArrMin = getArrRotMinutes(f);
      const vttDepMin = (typeof getDepBlockOutMin === 'function') ? getDepBlockOutMin(f) : 0;
      if (arrRwy === rwy.id) events.push({ time: sldtMin, type: 'arr', flight: f, cat: cat, vttArrMin, rotArrMin, index: eventIndex++ });
      if (depRwy === rwy.id) {
        events.push({ time: stotMin, type: 'dep', flight: f, cat: cat, vttDepMin, vttArrMin, rotArrMin, sobtMin: sobtMin, index: eventIndex++ });
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

