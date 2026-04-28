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

  function kpiDisposeInteractiveCharts() {
    try {
      if (window.__kpiChartGate) { window.__kpiChartGate.destroy(); window.__kpiChartGate = null; }
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
      const sibt = kpiToNumber(f && f.sibtMin_orig != null ? f.sibtMin_orig : (f && f.timeMin != null ? f.timeMin : null));
      const sldt = kpiToNumber(f && f.sldtMin_orig != null ? f.sldtMin_orig : (sibt != null && arrTaxiSch != null && rotSec != null ? Math.max(0, sibt - arrTaxiSch - rotSec / 60) : null));
      const dwellMin = kpiToNumber(f && f.dwellMin != null ? f.dwellMin : null);
      const sobt = kpiToNumber(f && f.sobtMin_orig != null ? f.sobtMin_orig : (sibt != null && dwellMin != null ? sibt + dwellMin : null));
      const sttDepMinK = kpiToNumber(typeof getBaseVttDepMinutesToHoldingSlot === 'function' ? getBaseVttDepMinutesToHoldingSlot(f) : depTaxiSch);
      const depRotMinK = depRotSec != null && isFinite(depRotSec) ? depRotSec / 60 : null;
      const stot = kpiToNumber(f && f.stotMin_orig != null ? f.stotMin_orig : (sobt != null && depRotMinK != null && sttDepMinK != null ? sobt + depRotMinK + sttDepMinK : (sobt != null && depBlockOutMin != null ? sobt + depBlockOutMin : null)));
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
  }
  function taxiSegmentVelocityMsForPolylineSegment(p1, p2, carry) {
    const mx = (p1[0] + p2[0]) * 0.5, my = (p1[1] + p2[1]) * 0.5;
    const hit = taxiHitFromMidpoint([mx, my]);
    return taxiSegmentVelocityMsFromHit(hit, carry);
  }
  function makeTaxiSegmentVelocityCallback() {
    const carry = { lastTaxiwayMs: null };
    return function(i, a, b) { return taxiSegmentVelocityMsForPolylineSegment(a, b, carry); };
  }
  function polylineRawDurationSegmentVelocities(pts, velForSeg) {
    if (!pts || pts.length < 2) return 0;
    let total = 0;
    for (let i = 0; i < pts.length - 1; i++) {
      const len = pathDist(pts[i], pts[i + 1]);
      if (len < 1e-9) continue;
      const v = Math.max(1, velForSeg(i, pts[i], pts[i + 1]));
      total += len / v;
    }
    return total;
  }
  function polylineTimelineBySegmentSpeeds(pts, tStart, tEnd, velForSeg) {
    if (!pts || pts.length < 2 || tEnd <= tStart + 1e-9) {
      const p = pts && pts.length ? pts[0] : [0, 0];
      return [{ t: tStart, x: p[0], y: p[1] }];
    }
    const lengths = [];
    for (let i = 0; i < pts.length - 1; i++) lengths.push(pathDist(pts[i], pts[i + 1]));
    const rawDts = [];
    for (let i = 0; i < lengths.length; i++) {
      const v = Math.max(1, velForSeg(i, pts[i], pts[i + 1]));
      rawDts.push((lengths[i] < 1e-9 ? 0 : lengths[i] / v));
    }
    const rawTotal = rawDts.reduce(function(s, x) { return s + x; }, 0);
    const window = tEnd - tStart;
    if (rawTotal < 1e-9) {
      return [
        { t: tStart, x: pts[0][0], y: pts[0][1] },
        { t: tEnd, x: pts[pts.length - 1][0], y: pts[pts.length - 1][1] },
      ];
    }
    const scale = window / rawTotal;
    const tl = [{ t: tStart, x: pts[0][0], y: pts[0][1] }];
    let acc = 0;
    for (let i = 0; i < lengths.length; i++) {
      acc += rawDts[i] * scale;
      tl.push({ t: Math.min(tStart + acc, tEnd), x: pts[i + 1][0], y: pts[i + 1][1] });
    }
    tl[tl.length - 1].t = tEnd;
    return tl;
  }
  function polylineTimelineConstantAccelFromRest(pts, tStart, tEnd, accelMs2) {
    if (!pts || pts.length < 2 || tEnd <= tStart + 1e-9) {
      const p = pts && pts.length ? polylinePointAtDistance(pts, 0) : [0, 0];
      return [{ t: tStart, x: p[0], y: p[1] }, { t: tEnd, x: p[0], y: p[1] }];
    }
    const L = polylineTotalLength(pts);
    const a = Math.max(0.1, accelMs2);
    const tPhys = L < 1e-9 ? 0 : Math.sqrt(2 * L / a);
    const win = tEnd - tStart;
    const n = Math.max(8, Math.min(48, Math.ceil(Math.max(L, 1) / 25)));
    const tl = [];
    for (let i = 0; i <= n; i++) {
      const u = i / n;
      const tt = tStart + u * win;
      const tau = u * tPhys;
      const s = Math.min(L, 0.5 * a * tau * tau);
      const pt = polylinePointAtDistance(pts, s);
      tl.push({ t: tt, x: pt[0], y: pt[1] });
    }
    tl[0].t = tStart;
    tl[tl.length - 1].t = tEnd;
    return tl;
  }
  function polylineTimelineLinearRetSpeed(pts, tStart, tEnd, vIn, vOut) {
    if (!pts || pts.length < 2 || tEnd <= tStart + 1e-9) {
      const p = pts && pts.length ? pts[0] : [0, 0];
      return [{ t: tStart, x: p[0], y: p[1] }];
    }
    const lengths = [];
    let totalLen = 0;
    for (let i = 0; i < pts.length - 1; i++) {
      const len = pathDist(pts[i], pts[i + 1]);
      lengths.push(len);
      totalLen += len;
    }
    const rawDts = [];
    let accLen = 0;
    for (let i = 0; i < lengths.length; i++) {
      const midLen = accLen + lengths[i] * 0.5;
      const u = totalLen > 1e-9 ? midLen / totalLen : 0;
      const v = Math.max(1, vIn + (vOut - vIn) * u);
      rawDts.push(lengths[i] < 1e-9 ? 0 : lengths[i] / v);
      accLen += lengths[i];
    }
