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
      const stot = kpiToNumber(f && f.stotMin_orig != null ? f.stotMin_orig : (sobt != null && depBlockOutMin != null ? sobt + depBlockOutMin : null));
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
