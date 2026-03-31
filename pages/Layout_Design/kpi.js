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
      const failed = !!(f && (f.noWayArr || f.noWayDep || f.arrRetFailed));
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
      host.innerHTML = '<div class="kpi-empty-state">No flights are available yet. Add or load a schedule, then click <strong>Light Sim</strong> to refresh the KPI snapshot.</div>';
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
        kpiBuildMetricRow('Snapshot basis', kpiFormatCount(snapshot.totalFlights) + ' flights', 'Rendered only on initial load and Light Sim')
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
