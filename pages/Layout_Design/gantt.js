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
        const flightId = ev.dataTransfer.getData('text/plain');
        if (!flightId) return;
        const f = st.flights.find(function(x) { return x.id === flightId; });
        if (!f) return;
        assignStandToFlight(f, track.getAttribute('data-stand-id') || null);
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
        ev.dataTransfer.setData('text/plain', this.getAttribute('data-flight-id') || '');
        ev.dataTransfer.effectAllowed = 'move';
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
        ev.preventDefault();
        ev.dataTransfer.dropEffect = 'move';
      });
      track.addEventListener('drop', function(ev) {
        ev.preventDefault();
        if (this.getAttribute('data-runway-legend') === '1') return;
        const flightId = ev.dataTransfer.getData('text/plain');
        if (!flightId) return;
        const f = st.flights.find(function(x) { return x.id === flightId; });
        if (!f) return;
        assignStandToFlight(f, this.getAttribute('data-stand-id') || null);
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
