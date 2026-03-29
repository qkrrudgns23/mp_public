
  function collectKpiSnapshot() {
    const flights = Array.isArray(state.flights) ? state.flights.slice() : [];
    if (typeof assignLineupQueueRanksAll === 'function') assignLineupQueueRanksAll(flights);
    const rows = flights.map(function(f) {
      const arrTaxiMin = kpiToNumber(typeof getBaseVttArrMinutes === 'function' ? getBaseVttArrMinutes(f) : null);
      const depBlockOutMin = kpiToNumber(typeof getDepBlockOutMin === 'function' ? getDepBlockOutMin(f) : null);
      const depTaxiMin = kpiToNumber(typeof getBaseVttDepMinutesToHoldingSlot === 'function' ? getBaseVttDepMinutesToHoldingSlot(f) : (typeof getBaseVttDepMinutesToLineup === 'function' ? getBaseVttDepMinutesToLineup(f) : null));
      const rotSec = kpiToNumber(f && f.arrRotSec != null ? f.arrRotSec : (typeof getArrRotMinutes === 'function' ? getArrRotMinutes(f) * 60 : null));
      let depRotSec = null;
      if (f && f.arrDep === 'Dep' && typeof computeDepRotSecondsForFlight === 'function') {
        depRotSec = computeDepRotSecondsForFlight(f);
      } else if (typeof SCHED_DEP_ROT_MIN === 'number' && isFinite(SCHED_DEP_ROT_MIN)) {
        depRotSec = SCHED_DEP_ROT_MIN * 60;
      }
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
      host.innerHTML = '<div class="kpi-empty-state">No flights are available yet. Add or load a schedule, then click <strong>Update</strong> to refresh the KPI snapshot.</div>';
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
        kpiBuildMetricRow('Snapshot basis', kpiFormatCount(snapshot.totalFlights) + ' flights', 'Rendered only on initial load and Update')
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

  function computeScheduledDisplayTimes(flights) {
    if (!flights || !flights.length) return;
    flights.forEach(f => {
      if (f.noWayArr || f.noWayDep) return;
      f.vttADelayMin = 0;
      const tArrMin = f.timeMin != null ? f.timeMin : 0;
      let dwell = f.dwellMin != null ? f.dwellMin : 0;
      let minDwell = f.minDwellMin != null ? f.minDwellMin : 0;
      dwell = Math.max(SCHED_DWELL_FLOOR_MIN, dwell);
      minDwell = Math.max(SCHED_DWELL_FLOOR_MIN, minDwell);
      if (minDwell > dwell) minDwell = dwell;
      f.dwellMin = dwell;
      f.minDwellMin = minDwell;
      let vttArrMin = getBaseVttArrMinutes(f);
      const rotArrMin = getArrRotMinutes(f);
      const depBlockOutMin = (typeof getDepBlockOutMin === 'function') ? getDepBlockOutMin(f) : 0;
      const sldtOrig = Math.max(0, tArrMin - vttArrMin - rotArrMin);
      const sobtOrig = tArrMin + dwell;
      const stotOrig = sobtOrig + depBlockOutMin;
      f.sldtMin_orig = sldtOrig;
      f.sibtMin_orig = tArrMin;
      f.sobtMin_orig = sobtOrig;
      f.stotMin_orig = stotOrig;
      f.sldtMin_d = f.sldtMin_orig;
      f.sibtMin_d = tArrMin;
      f.sobtMin_d = sobtOrig;
      f.stotMin_d = stotOrig;
    });
    const standToFlights = {};
    flights.forEach(f => {
      if (f.noWayArr || f.noWayDep || !f.standId) return;
      const sid = f.standId;
      if (!standToFlights[sid]) standToFlights[sid] = [];
      standToFlights[sid].push(f);
    });
    Object.keys(standToFlights).forEach(standId => {
      const list = standToFlights[standId];
      list.sort((a, b) => (a.sibtMin_d != null ? a.sibtMin_d : 0) - (b.sibtMin_d != null ? b.sibtMin_d : 0));
      let prevSOBT = -1e9;
      list.forEach(f => {
        const depBlockOutMin = (typeof getDepBlockOutMin === 'function') ? getDepBlockOutMin(f) : 0;
        const sibt0 = (f.sibtMin_d != null ? f.sibtMin_d : 0);
        const overlap = Math.max(0, prevSOBT - sibt0);
        f.vttADelayMin = overlap;
        f.sibtMin_d = sibt0 + overlap;
        const dwell = f.dwellMin != null ? f.dwellMin : SCHED_DWELL_FLOOR_MIN;
        const minDwell = f.minDwellMin != null ? f.minDwellMin : SCHED_DWELL_FLOOR_MIN;
        const minSobtByDwell = f.sibtMin_d + minDwell;
        const sobtCandidate = (f.sobtMin_d != null ? f.sobtMin_d : (f.sibtMin_d + dwell));
        f.sobtMin_d = Math.max(sobtCandidate, minSobtByDwell);
        f.stotMin_d = f.sobtMin_d + depBlockOutMin;
        prevSOBT = f.sobtMin_d;
      });
    });
    flights.forEach(f => {
      if (!f || f.noWayArr || f.noWayDep || !f.standId) return;
      const dwell = f.dwellMin != null ? f.dwellMin : SCHED_DWELL_FLOOR_MIN;
      const minDwell = f.minDwellMin != null ? f.minDwellMin : SCHED_DWELL_FLOOR_MIN;
      const sibt = (f.sibtMin_d != null ? f.sibtMin_d
                   : (f.sibtMin_orig != null ? f.sibtMin_orig : 0));
      const minSobtByDwell = sibt + minDwell;
      const sobtCurrent = (f.sobtMin_d != null ? f.sobtMin_d : (sibt + dwell));
      if (sobtCurrent < minSobtByDwell) {
        const delta = minSobtByDwell - sobtCurrent;
        f.sobtMin_d = minSobtByDwell;
        if (typeof f.stotMin_d === 'number') f.stotMin_d += delta;
      }
    });
    flights.forEach(f => {
      if (f.noWayArr || f.noWayDep) return;
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
      if (!f || f.noWayArr || f.noWayDep) return;
      if (f.standId && standsToRecompute.has(f.standId)) needStep1.add(f.id);
    });
    allFlights.forEach(function(f) {
      if (!f || !needStep1.has(f.id)) return;
      if (f.noWayArr || f.noWayDep) return;
      f.vttADelayMin = 0;
      const tArrMin = f.timeMin != null ? f.timeMin : 0;
      let dwell = f.dwellMin != null ? f.dwellMin : 0;
      let minDwell = f.minDwellMin != null ? f.minDwellMin : 0;
      dwell = Math.max(SCHED_DWELL_FLOOR_MIN, dwell);
      minDwell = Math.max(SCHED_DWELL_FLOOR_MIN, minDwell);
      if (minDwell > dwell) minDwell = dwell;
      f.dwellMin = dwell;
      f.minDwellMin = minDwell;
      const vttArrMin = getBaseVttArrMinutes(f);
      const rotArrMin = getArrRotMinutes(f);
      const depBlockOutMin = (typeof getDepBlockOutMin === 'function') ? getDepBlockOutMin(f) : 0;
      const sldtOrig = Math.max(0, tArrMin - vttArrMin - rotArrMin);
      const sobtOrig = tArrMin + dwell;
      const stotOrig = sobtOrig + depBlockOutMin;
      f.sldtMin_orig = sldtOrig;
      f.sibtMin_orig = tArrMin;
      f.sobtMin_orig = sobtOrig;
      f.stotMin_orig = stotOrig;
      f.sldtMin_d = f.sldtMin_orig;
      f.sibtMin_d = tArrMin;
      f.sobtMin_d = sobtOrig;
      f.stotMin_d = stotOrig;
    });
    standsToRecompute.forEach(function(standId) {
      const list = allFlights.filter(function(f) {
        return f && !f.noWayArr && !f.noWayDep && f.standId === standId;
      });
      list.sort((a, b) => (a.sibtMin_d != null ? a.sibtMin_d : 0) - (b.sibtMin_d != null ? b.sibtMin_d : 0));
      let prevSOBT = -1e9;
      list.forEach(function(f) {
        const depBlockOutMin = (typeof getDepBlockOutMin === 'function') ? getDepBlockOutMin(f) : 0;
        const sibt0 = (f.sibtMin_d != null ? f.sibtMin_d : 0);
        const overlap = Math.max(0, prevSOBT - sibt0);
        f.vttADelayMin = overlap;
        f.sibtMin_d = sibt0 + overlap;
        const dwell = f.dwellMin != null ? f.dwellMin : SCHED_DWELL_FLOOR_MIN;
        const minDwell = f.minDwellMin != null ? f.minDwellMin : SCHED_DWELL_FLOOR_MIN;
        const minSobtByDwell = f.sibtMin_d + minDwell;
        const sobtCandidate = (f.sobtMin_d != null ? f.sobtMin_d : (f.sibtMin_d + dwell));
        f.sobtMin_d = Math.max(sobtCandidate, minSobtByDwell);
        f.stotMin_d = f.sobtMin_d + depBlockOutMin;
        prevSOBT = f.sobtMin_d;
      });
    });
    allFlights.forEach(function(f) {
      if (!f || f.noWayArr || f.noWayDep || !f.standId) return;
      if (!standsToRecompute.has(f.standId)) return;
      const dwell = f.dwellMin != null ? f.dwellMin : SCHED_DWELL_FLOOR_MIN;
      const minDwell = f.minDwellMin != null ? f.minDwellMin : SCHED_DWELL_FLOOR_MIN;
      const sibt = (f.sibtMin_d != null ? f.sibtMin_d : (f.sibtMin_orig != null ? f.sibtMin_orig : 0));
      const minSobtByDwell = sibt + minDwell;
      const sobtCurrent = (f.sobtMin_d != null ? f.sobtMin_d : (sibt + dwell));
      if (sobtCurrent < minSobtByDwell) {
        const delta = minSobtByDwell - sobtCurrent;
        f.sobtMin_d = minSobtByDwell;
        if (typeof f.stotMin_d === 'number') f.stotMin_d += delta;
      }
    });
    allFlights.forEach(function(f) {
      if (!f || f.noWayArr || f.noWayDep) return;
      const onTouched = f.standId && standsToRecompute.has(f.standId);
      if (!needStep1.has(f.id) && !onTouched) return;
      f.sldtMin = f.sldtMin_d;
      f.stotMin = f.stotMin_d;
      f.sobtMin = f.sobtMin_d;
    });
  }