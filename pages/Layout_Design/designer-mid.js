      }
    });
  }
  const holdingPointNameInput = document.getElementById('holdingPointName');
  if (holdingPointNameInput) {
    holdingPointNameInput.addEventListener('change', function() {
      if (state.selectedObject && state.selectedObject.type === 'holdingPoint') {
        const hp = state.selectedObject.obj;
        const raw = (this.value || '').trim();
        if (raw && findDuplicateLayoutName('holdingPoint', hp.id, raw)) {
          alertDuplicateLayoutName();
          this.value = hp.name || '';
          return;
        }
        hp.name = raw;
        updateObjectInfo();
        renderObjectList();
        draw();
      }
    });
  }
  const remoteIcaoCategoriesHost = document.getElementById('remoteIcaoCategories');
  if (remoteIcaoCategoriesHost) {
    remoteIcaoCategoriesHost.addEventListener('change', function(ev) {
      const t = ev.target;
      if (!t || !t.classList.contains('icao-letter-check')) return;
      let letters = readIcaoCategoriesFromHost('remoteIcaoCategories');
      if (!letters.length) {
        letters = ['C'];
        applyIcaoCategoriesToHost('remoteIcaoCategories', letters);
      }
      const typeIds = aircraftTypeIdsForIcaoLetters(letters);
      if (state.selectedObject && state.selectedObject.type === 'remote') {
        const st = state.selectedObject.obj;
        st.categoryMode = 'icao';
        st.allowedIcaoCategories = letters;
        st.category = representativeCategoryFromLetters(letters);
        st.allowedAircraftTypes = typeIds;
        renderAircraftConstraintChoices('remoteAircraftAccess', typeIds, letters);
        updateObjectInfo();
        renderObjectList();
        draw();
        update3DSceneWhenVisible();
      } else {
        renderAircraftConstraintChoices('remoteAircraftAccess', typeIds, letters);
      }
    });
  }

  const remoteTerminalAccessEl = document.getElementById('remoteTerminalAccess');
  if (remoteTerminalAccessEl) {
    remoteTerminalAccessEl.addEventListener('change', function(ev) {
      const target = ev.target;
      if (!target || !target.classList.contains('remote-term-check')) return;
      syncChoiceChipStates(remoteTerminalAccessEl);
      if (!state.selectedObject || state.selectedObject.type !== 'remote') return;
      const st = state.selectedObject.obj;
      const checks = remoteTerminalAccessEl.querySelectorAll('.remote-term-check');
      const allowed = [];
      checks.forEach(function(ch) {
        if (ch.checked) {
          const id = ch.getAttribute('data-item-id');
          if (id) allowed.push(id);
        }
      });
      st.allowedTerminals = allowed;
      if (typeof syncPanelFromState === 'function') syncPanelFromState();
      updateObjectInfo();
      renderObjectList();
      draw();
    });
  }
  const remoteAircraftAccessEl = document.getElementById('remoteAircraftAccess');
  if (remoteAircraftAccessEl) {
    remoteAircraftAccessEl.addEventListener('change', function(ev) {
      const target = ev.target;
      if (!target || !target.classList.contains('aircraft-type-check')) return;
      syncChoiceChipStates(remoteAircraftAccessEl);
      if (!state.selectedObject || state.selectedObject.type !== 'remote') return;
      const stAc = state.selectedObject.obj;
      applyUnifiedStandConstraintFromPanelToObject(stAc, 'remoteIcaoCategories', 'remoteAircraftAccess');
      renderAircraftConstraintChoices('remoteAircraftAccess', stAc.allowedAircraftTypes, stAc.allowedIcaoCategories);
      updateObjectInfo();
      renderObjectList();
      draw();
    });
  }
  const tempStandNameInput = document.getElementById('tempStandName');
  if (tempStandNameInput) {
    tempStandNameInput.addEventListener('change', function() {
      if (state.selectedObject && state.selectedObject.type === 'tempStand') {
        const st = state.selectedObject.obj;
        const raw = (this.value || '').trim();
        if (raw && findDuplicateLayoutName('tempStand', st.id, raw)) {
          alertDuplicateLayoutName();
          this.value = st.name || '';
          return;
        }
        st.name = raw;
        updateObjectInfo();
        renderObjectList();
        draw();
        update3DSceneWhenVisible();
      }
    });
  }
  const tempStandIcaoCategoriesHost = document.getElementById('tempStandIcaoCategories');
  if (tempStandIcaoCategoriesHost) {
    tempStandIcaoCategoriesHost.addEventListener('change', function(ev) {
      const t = ev.target;
      if (!t || !t.classList.contains('icao-letter-check')) return;
      let letters = readIcaoCategoriesFromHost('tempStandIcaoCategories');
      if (!letters.length) {
        letters = ['C'];
        applyIcaoCategoriesToHost('tempStandIcaoCategories', letters);
      }
      const typeIds = aircraftTypeIdsForIcaoLetters(letters);
      if (state.selectedObject && state.selectedObject.type === 'tempStand') {
        const st = state.selectedObject.obj;
        st.categoryMode = 'icao';
        st.allowedIcaoCategories = letters;
        st.category = representativeCategoryFromLetters(letters);
        st.allowedAircraftTypes = typeIds;
        renderAircraftConstraintChoices('tempStandAircraftAccess', typeIds, letters);
        updateObjectInfo();
        renderObjectList();
        draw();
        update3DSceneWhenVisible();
      } else {
        renderAircraftConstraintChoices('tempStandAircraftAccess', typeIds, letters);
      }
    });
  }
  const tempStandTerminalAccessEl = document.getElementById('tempStandTerminalAccess');
  if (tempStandTerminalAccessEl) {
    tempStandTerminalAccessEl.addEventListener('change', function(ev) {
      const target = ev.target;
      if (!target || !target.classList.contains('remote-term-check')) return;
      syncChoiceChipStates(tempStandTerminalAccessEl);
      if (!state.selectedObject || state.selectedObject.type !== 'tempStand') return;
      const st = state.selectedObject.obj;
      const checks = tempStandTerminalAccessEl.querySelectorAll('.remote-term-check');
      const allowed = [];
      checks.forEach(function(ch) {
        if (ch.checked) {
          const id = ch.getAttribute('data-item-id');
          if (id) allowed.push(id);
        }
      });
      st.allowedTerminals = allowed;
      if (typeof syncPanelFromState === 'function') syncPanelFromState();
      updateObjectInfo();
      renderObjectList();
      draw();
    });
  }
  const tempStandAircraftAccessEl = document.getElementById('tempStandAircraftAccess');
  if (tempStandAircraftAccessEl) {
    tempStandAircraftAccessEl.addEventListener('change', function(ev) {
      const target = ev.target;
      if (!target || !target.classList.contains('aircraft-type-check')) return;
      syncChoiceChipStates(tempStandAircraftAccessEl);
      if (!state.selectedObject || state.selectedObject.type !== 'tempStand') return;
      const tstAc = state.selectedObject.obj;
      applyUnifiedStandConstraintFromPanelToObject(tstAc, 'tempStandIcaoCategories', 'tempStandAircraftAccess');
      renderAircraftConstraintChoices('tempStandAircraftAccess', tstAc.allowedAircraftTypes, tstAc.allowedIcaoCategories);
      updateObjectInfo();
      renderObjectList();
      draw();
    });
  }

  document.getElementById('taxiwayName').addEventListener('change', function() {
    if (state.selectedObject && state.selectedObject.type === 'taxiway') {
      const tw = state.selectedObject.obj;
      const raw = (this.value || '').trim();
      if (raw && findDuplicateLayoutName('taxiway', tw.id, raw)) {
        alertDuplicateLayoutName();
        this.value = tw.name || '';
        return;
      }
      tw.name = raw;
      updateObjectInfo();
      renderObjectList();
      draw();
    }
  });
  const apronLinkNameInputEl = document.getElementById('apronLinkName');
  if (apronLinkNameInputEl) {
    apronLinkNameInputEl.addEventListener('change', function() {
      if (state.selectedObject && state.selectedObject.type === 'apronLink') {
        const lk = state.selectedObject.obj;
        const rawTrim = (this.value || '').trim();
        const candidate = rawTrim || getApronLinkDefaultName(lk.id);
        if (findDuplicateLayoutName('apronLink', lk.id, candidate)) {
          alertDuplicateLayoutName();
          this.value = getApronLinkDisplayName(lk);
          return;
        }
        lk.name = rawTrim;
        this.value = getApronLinkDisplayName(lk);
        updateObjectInfo();
        renderObjectList();
        draw();
      }
    });
  }
  const edgeNameInputEl = document.getElementById('edgeName');
  if (edgeNameInputEl) {
    edgeNameInputEl.addEventListener('change', function() {
      if (state.selectedObject && state.selectedObject.type === 'layoutEdge') {
        const ed = state.selectedObject.obj;
        const rawTrim = (this.value || '').trim();
        const candidate = rawTrim || getLayoutEdgeDefaultName(ed);
        if (findDuplicateLayoutName('layoutEdge', ed.id, candidate)) {
          alertDuplicateLayoutName();
          this.value = getLayoutEdgeDisplayName(ed);
          return;
        }
        state.layoutEdgeNames[ed.id] = candidate;
        ed.name = candidate;
        this.value = candidate;
        updateObjectInfo();
        renderObjectList();
        draw();
      }
    });
  }
  document.getElementById('taxiwayWidth').addEventListener('change', function() {
    if (state.selectedObject && state.selectedObject.type === 'taxiway') {
      const tw = state.selectedObject.obj;
      const baseWidth = tw.pathType === 'runway'
        ? RUNWAY_PATH_DEFAULT_WIDTH
        : (tw.pathType === 'runway_exit' ? RUNWAY_EXIT_DEFAULT_WIDTH : TAXIWAY_DEFAULT_WIDTH);
      const val = Number(this.value);
      tw.width = clampTaxiwayWidthM(tw.pathType || 'taxiway', val, baseWidth);
      this.value = tw.width;
      updateObjectInfo();
      draw();
      update3DSceneWhenVisible();
    }
  });
  const pathPavementSel = document.getElementById('pathPavement');
  if (pathPavementSel) pathPavementSel.addEventListener('change', function() {
    if (state.selectedObject && state.selectedObject.type === 'taxiway') {
      const tw = state.selectedObject.obj;
      tw.pavement = getPathPavementFromPanelForPathType(tw.pathType || 'taxiway');
      updateObjectInfo();
      draw();
      update3DSceneWhenVisible();
    }
  });
  const avgVelInputEl = document.getElementById('taxiwayAvgMoveVelocity');
  if (avgVelInputEl) avgVelInputEl.addEventListener('change', function() {
    if (state.selectedObject && state.selectedObject.type === 'taxiway') {
      const tw = state.selectedObject.obj;
      const val = Number(this.value);
      const v =
        (typeof val === 'number' && isFinite(val) && val > 0)
          ? Math.max(1, Math.min(50, val))
          : 10;
      tw.avgMoveVelocity = v;
      this.value = v;
      updateObjectInfo();
      renderObjectList();
      draw();
      update3DSceneWhenVisible();
    }
  });
  document.getElementById('taxiwayMaxExitVel').addEventListener('change', function() {
    if (state.selectedObject && state.selectedObject.type === 'taxiway') {
      const tw = state.selectedObject.obj;
      const val = Number(this.value);
      if (tw.pathType === 'runway_exit') {
        tw.maxExitVelocity = isFinite(val) && val > 0 ? val : null;
        if (typeof tw.minExitVelocity === 'number' && isFinite(tw.minExitVelocity) && tw.maxExitVelocity != null && tw.minExitVelocity > tw.maxExitVelocity) {
          tw.minExitVelocity = tw.maxExitVelocity;
        }
      } else {
        delete tw.maxExitVelocity;
      }
      if (isFinite(val) && val > 0) this.value = val; else this.value = tw.maxExitVelocity != null ? tw.maxExitVelocity : '';
      updateObjectInfo();
      renderObjectList();
      draw();
      update3DSceneWhenVisible();
    }
  });
  const minExitEl = document.getElementById('taxiwayMinExitVel');
  if (minExitEl) {
    minExitEl.addEventListener('change', function() {
      if (state.selectedObject && state.selectedObject.type === 'taxiway') {
        const tw = state.selectedObject.obj;
        const val = Number(this.value);
        if (tw.pathType === 'runway_exit') {
          let v = isFinite(val) && val > 0 ? val : 15;
          if (typeof tw.maxExitVelocity === 'number' && isFinite(tw.maxExitVelocity) && v > tw.maxExitVelocity) v = tw.maxExitVelocity;
          tw.minExitVelocity = v;
          this.value = v;
        } else {
          delete tw.minExitVelocity;
        }
        updateObjectInfo();
        renderObjectList();
        draw();
        update3DSceneWhenVisible();
      }
    });
  }
  const runwayExitAllowedDirectionEl = document.getElementById('runwayExitAllowedDirection');
  function triggerArrivalConfigResampleFromLayoutEdit() {
    if (typeof bumpVttArrCacheRev === 'function') bumpVttArrCacheRev();
    if (typeof bumpScheduleRetExitDistCache === 'function') bumpScheduleRetExitDistCache();
    if (typeof renderFlightList === 'function') renderFlightList(false, true);
  }
  if (runwayExitAllowedDirectionEl) {
    runwayExitAllowedDirectionEl.addEventListener('change', function(ev) {
      const target = ev.target;
      if (!target || !target.classList.contains('runway-exit-dir-check')) return;
      syncChoiceChipStates(runwayExitAllowedDirectionEl);
      if (!(state.selectedObject && state.selectedObject.type === 'taxiway')) return;
      const tw = state.selectedObject.obj;
      if (!tw || tw.pathType !== 'runway_exit') return;
      tw.allowedRwDirections = getRunwayExitAllowedDirectionsFromPanel();
        updateObjectInfo();
        renderObjectList();
        if (typeof redrawLayoutAfterEdit === 'function') redrawLayoutAfterEdit();
        else if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths();
        else draw();
      });
  }
    document.getElementById('taxiwayDirectionMode').addEventListener('change', function() {
    if (state.selectedObject && state.selectedObject.type === 'taxiway') {
      const tw = state.selectedObject.obj;
      const v = this.value || '';
      if (tw.pathType === 'runway') {
        runwayReverseVerticesIfDirectionChanged(tw, v);
        tw.direction = (v === 'counter_clockwise') ? 'counter_clockwise' : 'clockwise';
      } else tw.direction = v || 'both';
      updateObjectInfo();
      if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
      draw();
      update3DSceneWhenVisible();
    }
  });
  const taxiwayPathTypeKindEl = document.getElementById('taxiwayPathTypeKind');
  if (taxiwayPathTypeKindEl) {
    taxiwayPathTypeKindEl.addEventListener('change', function() {
      if (state.selectedObject && state.selectedObject.type === 'taxiway') {
        const tw = state.selectedObject.obj;
        const ptCur = tw.pathType || 'taxiway';
        if (ptCur === 'taxiway' || ptCur === 'general_queue_taxiway') {
          const kind = String(this.value || 'normal');
          tw.pathType = (kind === 'queue') ? 'general_queue_taxiway' : 'taxiway';
        }
        updateObjectInfo();
        if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
        draw();
        update3DSceneWhenVisible();
      }
    });
  }
  const runwayMinArrVelEl = document.getElementById('runwayMinArrVelocity');
  if (runwayMinArrVelEl) {
    runwayMinArrVelEl.addEventListener('change', function() {
      if (state.selectedObject && state.selectedObject.type === 'taxiway') {
        const tw = state.selectedObject.obj;
        if (tw.pathType !== 'runway') return;
        const val = Number(this.value);
        const v = (typeof val === 'number' && isFinite(val) && val > 0) ? Math.max(1, Math.min(150, val)) : 15;
        tw.minArrVelocity = v;
        this.value = v;
        updateObjectInfo();
        renderObjectList();
        if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
        draw();
      }
    });
  }
  [
    ['runwayLineupDistM_CW', 'clockwise'],
    ['runwayLineupDistM_CCW', 'counter_clockwise']
  ].forEach(function(item) {
    const lineupEl = document.getElementById(item[0]);
    if (!lineupEl) return;
    lineupEl.addEventListener('change', function() {
      if (state.selectedObject && state.selectedObject.type === 'taxiway') {
        const tw = state.selectedObject.obj;
        if (tw.pathType !== 'runway') return;
        const val = Number(this.value);
        const v = (typeof val === 'number' && isFinite(val) && val >= 0) ? val : 0;
        if (item[1] === 'clockwise') tw.lineupDistM_CW = v;
        else tw.lineupDistM_CCW = v;
        tw.lineupDistM = getEffectiveRunwayLineupDistM(tw);
        this.value = String(v);
        updateObjectInfo();
        if (typeof redrawLayoutAfterEdit === 'function') redrawLayoutAfterEdit();
        else if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths(); else draw();
      }
    });
  });
  [
    ['runwayStartDisplacedThresholdM', 'startDisplacedThresholdM', function(tw) { return getEffectiveRunwayStartDisplacedThresholdM(tw); }],
    ['runwayStartBlastPadM', 'startBlastPadM', function(tw) { return getEffectiveRunwayStartBlastPadM(tw); }],
    ['runwayEndDisplacedThresholdM', 'endDisplacedThresholdM', function(tw) { return getEffectiveRunwayEndDisplacedThresholdM(tw); }],
    ['runwayEndBlastPadM', 'endBlastPadM', function(tw) { return getEffectiveRunwayEndBlastPadM(tw); }]
  ].forEach(function(item) {
    const el = document.getElementById(item[0]);
    if (!el) return;
    el.addEventListener('change', function() {
      if (state.selectedObject && state.selectedObject.type === 'taxiway') {
        const tw = state.selectedObject.obj;
        if (tw.pathType !== 'runway') return;
        const val = Number(this.value);
        const v = (typeof val === 'number' && isFinite(val) && val >= 0) ? val : item[2](tw);
        tw[item[1]] = v;
        this.value = String(v);
        updateObjectInfo();
        draw();
        if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
      }
    });
  });

  function getMinArrVelocityMpsForRunwayId(runwayId) {
    if (runwayId == null || runwayId === '') return 15;
    const list = state.taxiways || [];
    let tw = list.find(t => t.id === runwayId && t.pathType === 'runway');
    if (!tw) return 15;
    const v = tw.minArrVelocity;
    if (typeof v === 'number' && isFinite(v) && v > 0) return Math.max(1, Math.min(150, v));
    return 15;
  }
  
  function runwayArrSpeedAndTimeToRet(v0, a, distM, vFloorIn) {
    const vf0 = Math.max(1, Math.min(150, vFloorIn));
    const vf = Math.min(vf0, v0);
    if (!(a > 0) || distM <= 0) return { vAtRet: v0, tSec: 0 };
    if (v0 <= vf) return { vAtRet: v0, tSec: distM / Math.max(v0, 1e-6) };
    const dStop = (v0 * v0 - vf * vf) / (2 * a);
    if (distM < dStop) {
      const vEnd = Math.sqrt(Math.max(0, v0 * v0 - 2 * a * distM));
      return { vAtRet: vEnd, tSec: (v0 - vEnd) / a };
    }
    const tDecel = (v0 - vf) / a;
    const tCruise = (distM - dStop) / vf;
    return { vAtRet: vf, tSec: tDecel + tCruise };
  }
  function parseTimeToMinutes(val) {
    if (!val) return 0;
    const s = String(val).trim();
    if (!s) return 0;
    const isoDt = s.match(/^(\d{4})-(\d{2})-(\d{2})[ T]+(\d{1,2}):(\d{2})(?::(\d{2}))?/);
    if (isoDt) {
      const h = parseInt(isoDt[4], 10) || 0;
      const m = parseInt(isoDt[5], 10) || 0;
      const sec = isoDt[6] ? (parseInt(isoDt[6], 10) || 0) : 0;
      return Math.max(0, h * 60 + m + sec / 60);
    }
    if (s.includes(':')) {
      const parts = s.split(':');
      const h = parseInt(parts[0], 10) || 0;
      const m = parseInt(parts[1], 10) || 0;
      const sec = (parts.length >= 3) ? (parseInt(parts[2], 10) || 0) : 0;
      return Math.max(0, h * 60 + m + sec / 60);
    }
    const num = parseFloat(s);
    return isNaN(num) ? 0 : Math.max(0, num);
  }

  function snapSimTimeSecForSlider(tSec) {
    const lo = state.simStartSec;
    const hi = state.simDurationSec;
    const step = SIM_TIME_SLIDER_SNAP_SEC;
    const t = Number(tSec);
    if (!isFinite(t)) return lo;
    if (!isFinite(lo) || !isFinite(hi) || hi < lo) return t;
    const clamped = Math.max(lo, Math.min(hi, t));
    if (!(step > 0)) return clamped;
    let snapped = lo + Math.round((clamped - lo) / step) * step;
    if (snapped < lo) snapped = lo;
    if (snapped > hi) snapped = hi;
    return snapped;
  }
  function updateFlightSimPlaybackLabelsDom() {
    const label = document.getElementById('flightSimTimeLabel');
    const t = state.simTimeSec;
    if (label) label.textContent = formatSecondsToHHMMSS(t);
  }
  
  function minFirstArrivalTouchdownSecAmongFlights() {
    let minS = Infinity;
    (state.flights || []).forEach(function(f) {
      if (!f || f.arrDep === 'Dep') return;
      if (arrivalAirsideBlocked(f)) return;
      const w = getFlightAirsideWindowSec(f);
      if (!w) return;
      const eldtMin = flightEMinutesPrefer(f, ['eldtMin'], flightEMinutesPrefer(f, ['timeMin'], NaN));
      if (!isFinite(eldtMin)) return;
      const eldtS = eldtMin * 60;
      if (eldtS < minS) minS = eldtS;
    });
    return (isFinite(minS) && minS < Infinity) ? minS : null;
  }
  function recomputeSimDuration() {
    let minT = Infinity;
    let maxT = -Infinity;
    (state.flights || []).forEach(function(f) {
      if (!f) return;
      const trWin = compactPlaybackTrackStartEnd(compactPlaybackTrackForFlight(f));
      if (trWin) {
        if (trWin.t0 < minT) minT = trWin.t0;
        if (trWin.t1 > maxT) maxT = trWin.t1;
      }
      const w = trWin ? null : getFlightAirsideWindowSec(f);
      if (w) {
        if (w.t0 < minT) minT = w.t0;
        if (w.t1 > maxT) maxT = w.t1;
      }
      const m = f.timeline_meta;
      const etotSec = m && typeof m.etotSec === 'number' ? Number(m.etotSec) : NaN;
      if (isFinite(etotSec) && etotSec > maxT) maxT = etotSec;
    });
    if (!isFinite(minT) || !isFinite(maxT)) {
      minT = 0;
      maxT = 0;
    }
    let simLo = minT;
    const firstTdS = minFirstArrivalTouchdownSecAmongFlights();
    if (firstTdS != null) {
      simLo = Math.max(0, firstTdS - 10);
    }
    let durSec = Math.max(maxT, minT);
    const capAbs = state.simPlaybackEndCapSec;
    if (capAbs != null && isFinite(Number(capAbs))) {
      durSec = Math.min(durSec, Number(capAbs));
    }
    state.simDurationSec = durSec;
    if (simLo > state.simDurationSec - 1e-6) {
      simLo = Math.max(0, state.simDurationSec - 1);
    }
    state.simStartSec = simLo;
    if ((state.flights || []).length > 0 && isFinite(minT) && isFinite(maxT) && state.simDurationSec <= state.simStartSec) {
      state.simDurationSec = state.simStartSec + 1;
    }
    state.simTimeSec = Math.max(state.simStartSec, Math.min(state.simDurationSec, state.simTimeSec));
    state.simTimeSec = snapSimTimeSecForSlider(state.simTimeSec);
    const slider = document.getElementById('flightSimSlider');
    if (slider) {
      slider.min = state.simStartSec;
      slider.max = state.simDurationSec;
      slider.step = String(SIM_TIME_SLIDER_SNAP_SEC);
      slider.value = state.simTimeSec;
      if (state.simDurationSec <= state.simStartSec) slider.disabled = true;
      else slider.disabled = false;
    }
    if (typeof renderFlightSimSliderDeadlockMarkers === 'function') renderFlightSimSliderDeadlockMarkers();
    updateFlightSimPlaybackLabelsDom();
    if (typeof applySimPlaybackBarDomVisibility === 'function') applySimPlaybackBarDomVisibility();
  }
  function applySimPlaybackBarDomVisibility() {
    const wrap = document.getElementById('sim-controls-wrap');
    const inner = document.getElementById('sim-controls-container');
    const hideBtn = document.getElementById('btnHideSimPlaybackBar');
    const hasSim = state.hasSimulationResult && state.flights.length > 0;
    if (!wrap) return;
    if (!hasSim || !state.simPlaybackDockVisible) {
      wrap.style.display = 'none';
      return;
    }
    wrap.style.display = 'flex';
    if (inner) inner.style.display = 'flex';
    if (hideBtn) hideBtn.setAttribute('aria-expanded', 'true');
  }
  function syncSimulationPlaybackAfterTimelines() {
    if (typeof recomputeSimDuration === 'function') recomputeSimDuration();
    if (!state.hasSimulationResult) return;
    const simSliderAfter = document.getElementById('flightSimSlider');
    state.simTimeSec = snapSimTimeSecForSlider(Math.max(state.simStartSec, Math.min(state.simDurationSec, state.simStartSec)));
    if (simSliderAfter) simSliderAfter.value = state.simTimeSec;
    updateFlightSimPlaybackLabelsDom();
  }

  function formatTotalSecondsToHHMMSS(totalSec) {
    const parts = _splitTotalSeconds(totalSec);
    return parts.hh + ':' + parts.mm + ':' + parts.ss;
  }
  function formatMinutesToHHMMSS(minsRaw) {
    return formatTotalSecondsToHHMMSS(_normalizeTimeToSeconds(minsRaw, 'minutes', 'round'));
  }
  function flightScheduleBaseDateIso(f) {
    if (!f) return DEFAULT_SIBT_DATE;
    const raw = f.sibtDate != null ? f.sibtDate : (f.serviceDate != null ? f.serviceDate : null);
    const d = (raw == null ? '' : String(raw)).trim();
    if (/^\d{4}-\d{2}-\d{2}$/.test(d)) return d;
    return DEFAULT_SIBT_DATE;
  }
  function formatFlightScheduleDateTime(f, minsRaw) {
    const base = flightScheduleBaseDateIso(f);
    const sec = _normalizeTimeToSeconds(minsRaw, 'minutes', 'round');
    const minTotal = sec / 60;
    const ps = base.split('-');
    const Y = parseInt(ps[0], 10);
    const Mo = parseInt(ps[1], 10) - 1;
    const D = parseInt(ps[2], 10);
    if (!isFinite(Y) || !isFinite(Mo) || !isFinite(D)) return formatMinutesToHHMMSS(minsRaw);
    const t0 = new Date(Y, Mo, D, 0, 0, 0);
    t0.setMinutes(t0.getMinutes() + minTotal);
    const pad = function(n) { return (n < 10 ? '0' : '') + n; };
    return t0.getFullYear() + '-' + pad(t0.getMonth() + 1) + '-' + pad(t0.getDate()) + ' ' + pad(t0.getHours()) + ':' + pad(t0.getMinutes()) + ':' + pad(t0.getSeconds());
  }
  function formatSignedMinutesToHHMMSS(minsRaw) {
    const n = Number(minsRaw);
    if (!isFinite(n)) return '—';
    const sign = n < 0 ? '-' : '';
    return sign + formatMinutesToHHMMSS(Math.abs(n));
  }
  function formatSecondsToHHMMSS(secRaw) {
    return formatTotalSecondsToHHMMSS(_normalizeTimeToSeconds(secRaw, 'seconds', 'floor'));
  }

  function getStandBusyIntervals(standId, ignoreFlightId) {
    const intervals = [];
    if (!standId) return intervals;
    (state.flights || []).forEach(f => {
      if (!f || f.id === ignoreFlightId) return;
      if (f.arrDep !== 'Arr') return;
      if (f.standId !== standId) return;
      const win = getFlightAirsideWindowSec(f);
      if (!win) return;
      const end = win.t1;
      const dwellMin = (f.sobtMin != null && f.sibtMin != null) ? (f.sobtMin - f.sibtMin) : (f.dwellMin || 0);
      const dwellSec = Math.max(0, dwellMin * 60);
      const start = Math.max(0, end - dwellSec);
      if (end > start) intervals.push({ start, end });
    });
    intervals.sort((a, b) => a.start - b.start);
    return intervals;
  }

  function isStandOccupiedAtSimSec(standId, tSec) {
    if (!standId || !simPlaybackVisualsActive() || simPlaybackHeavyVisualsSuppressed()) return false;
    const t = Number(tSec);
    if (!isFinite(t)) return false;
    return getSimStandOccupancySetAtSec(t).has(String(standId));
  }

  function getSimStandOccupancySetAtSec(tSec) {
    const t = Number(tSec);
    const empty = new Set();
    if (!simPlaybackVisualsActive() || simPlaybackHeavyVisualsSuppressed() || !isFinite(t)) return empty;
    const key = String((state.pathPolylineCacheRev | 0)) + '|' + String((state.flights || []).length) + '|' + t.toFixed(3);
    if (_simStandOccupancyCache && _simStandOccupancyCache.key === key) return _simStandOccupancyCache.set;
    const set = new Set();
    const flights = state.flights || [];
    for (let i = 0; i < flights.length; i++) {
      const f = flights[i];
      if (!f || !f.standId) continue;
      const m = f.timeline_meta;
      if (m && typeof m.eibtSec === 'number' && typeof m.eobtSec === 'number') {
        if (t + 1e-3 >= m.eibtSec && t <= m.eobtSec + 1e-3) set.add(String(f.standId));
        continue;
      }
      if (f.arrDep !== 'Dep' && (f.noWayArr || f.arrRetFailed)) {
        const eldtMin = flightEMinutesPrefer(f, ['eldtMin'], flightEMinutesPrefer(f, ['timeMin'], 0));
        const eibtMin = flightEMinutesPrefer(f, ['eibtMin'], eldtMin + 15);
        const eobtMin = flightEMinutesPrefer(f, ['eobtMin'], eibtMin + (typeof f.dwellMin === 'number' && isFinite(f.dwellMin) ? f.dwellMin : 45));
        const eibtS = eibtMin * 60;
        const eobtS = eobtMin * 60;
        if (t + 1e-3 >= eibtS && t <= eobtS + 1e-3) set.add(String(f.standId));
      }
    }
    _simStandOccupancyCache = { key: key, set: set };
    return set;
  }

  function findStandAvailableArrivalTime(standId, desiredArrival, dwellSec) {
    let s = Math.max(0, desiredArrival);
    const intervals = getStandBusyIntervals(standId, null);
    for (let i = 0; i < intervals.length; i++) {
      const iv = intervals[i];
      if (s + dwellSec <= iv.start) return s;
      if (s < iv.end) s = iv.end;
    }
    return s;
  }

  function getTerminalForStand(stand) {
    if (!stand || !state.terminals.length) return null;
    const [px, py] = getStandConnectionPx(stand);
    let nearest = null;
    let nearestD2 = Infinity;
    for (let i = 0; i < state.terminals.length; i++) {
      const t = state.terminals[i];
      if (!t.vertices || t.vertices.length < 1) continue;
      const termPix = t.vertices.map(v => cellToPixel(v.col, v.row));
      if (t.closed && termPix.length >= 3 && pointInPolygonXY([px, py], termPix)) return t;
      let cx = 0, cy = 0;
      termPix.forEach(p => { cx += p[0]; cy += p[1]; });
      cx /= termPix.length;
      cy /= termPix.length;
      const dx = px - cx, dy = py - cy;
      const d2 = dx*dx + dy*dy;
      if (d2 < nearestD2) {
        nearestD2 = d2;
        nearest = t;
      }
    }
    return nearest;
  }

  function allStandsForFlightAssignment() {
    return (state.pbbStands || []).concat(state.remoteStands || []).concat(state.tempStands || []);
  }

  function flightStandAircraftConstraintOk(f, stand) {
    if (!stand) return true;
    const mode = getStandCategoryMode(stand);
    const allowedTypes = getStandAllowedAircraftTypes(stand);
    if (allowedTypes.length) {
      const flightType = String(f.aircraftType || '').trim();
      if (!flightType || allowedTypes.indexOf(flightType) < 0) return false;
    } else if (mode === 'aircraft') {
      return false;
    } else {
      const order = { A:1,B:2,C:3,D:4,E:5,F:6 };
      const fCode = String(f.code || 'C').toUpperCase()[0];
      const sCat = String(stand.category || 'F').toUpperCase()[0];
      const fc = order[fCode] || 99;
      const sc = order[sCat] || 0;
      if (fc > sc) return false;
    }
    return true;
  }
  function standCanUseTerminalForFlight(stand, terminalId) {
    if (!stand) return true;
    const ft = terminalId || null;
    if (!ft) return true;
    const isRemoteLike = (state.remoteStands || []).some(function(r) { return r.id === stand.id; })
      || (state.tempStands || []).some(function(r) { return r.id === stand.id; });
    if (isRemoteLike) {
      const allowed = Array.isArray(stand.allowedTerminals) ? stand.allowedTerminals : [];
      if (allowed.length) return allowed.indexOf(ft) >= 0;
    }
    const term = getTerminalForStand(stand);
    const standTermId = term ? term.id : null;
    if (!standTermId) return false;
    return ft === standTermId;
  }
  function flightSegmentTerminalIdForValidation(f, segmentIdx, segmentCount) {
    const count = Math.max(1, Number(segmentCount) || 1);
    const idx = Math.max(0, Number(segmentIdx) || 0);
    if (idx === 0) return resolveFlightArrTerminalId(f);
    if (idx >= count - 1) return resolveFlightDepTerminalId(f);
    return null;
  }
  function flightCanUseStandForSegment(f, stand, segmentIdx, segmentCount) {
    if (!flightStandAircraftConstraintOk(f, stand)) return false;
    const count = Math.max(1, Number(segmentCount) || 1);
    if (count === 1) {
      const arrTermId = resolveFlightArrTerminalId(f);
      const depTermId = resolveFlightDepTerminalId(f);
      if (!standCanUseTerminalForFlight(stand, arrTermId)) return false;
      if (depTermId && depTermId !== arrTermId && !standCanUseTerminalForFlight(stand, depTermId)) return false;
      return true;
    }
    const termId = flightSegmentTerminalIdForValidation(f, segmentIdx, segmentCount);
    return standCanUseTerminalForFlight(stand, termId);
  }
  function flightCanUseStand(f, stand) {
    const segs = typeof normalizeFlightApronStaySegments === 'function' ? normalizeFlightApronStaySegments(f) : [];
    return flightCanUseStandForSegment(f, stand, 0, Math.max(1, segs.length || 1));
  }
  function showAllocationConstraintModal(message) {
    const msg = String(message || 'This stand assignment is not allowed.');
    let el = document.getElementById('allocConstraintModal');
    if (!el) {
      el = document.createElement('div');
      el.id = 'allocConstraintModal';
      el.className = 'alloc-constraint-modal';
      el.innerHTML = '<div class="alloc-constraint-modal__panel" role="alertdialog" aria-modal="true"><div class="alloc-constraint-modal__title">Assignment not allowed</div><div class="alloc-constraint-modal__message"></div><button type="button" class="alloc-constraint-modal__button">OK</button></div>';
      document.body.appendChild(el);
      const btn = el.querySelector('.alloc-constraint-modal__button');
      if (btn) btn.addEventListener('click', function() { el.classList.remove('is-open'); });
      el.addEventListener('click', function(ev) { if (ev.target === el) el.classList.remove('is-open'); });
    }
    const msgEl = el.querySelector('.alloc-constraint-modal__message');
    if (msgEl) msgEl.textContent = msg;
    el.classList.add('is-open');
  }

  function assignStandToFlight(f, standId, segmentIdx) {
    if (!f) return false;
    if (standId) {
      const allStands = allStandsForFlightAssignment();
      const stand = allStands.find(function(s) { return s.id === standId; });
      const segsForValidation = typeof normalizeFlightApronStaySegments === 'function' ? normalizeFlightApronStaySegments(f) : [];
      const segCount = Math.max(1, segsForValidation.length || 1);
      const segIdxForValidation = segmentIdx != null && isFinite(Number(segmentIdx)) ? Math.max(0, parseInt(segmentIdx, 10) || 0) : 0;
      if (!flightCanUseStandForSegment(f, stand, segIdxForValidation, segCount)) {
        showAllocationConstraintModal("Stand constraints or selected Arr/Dep Building do not match this aircraft, so it cannot be assigned.");
        return false;
      }
      if (typeof computeScheduledDisplayTimes === 'function') computeScheduledDisplayTimes(state.flights);
      if (flightWouldOverlapStandAssignment(f, standId, segIdxForValidation)) {
        showAllocationConstraintModal("This stand or a safety-overlapped stand already has an overlapping flight in the selected SIBT-SOBT window.");
        return false;
      }
    }
    const prevStandForSched = f.standId || null;
    const segIdx = segmentIdx != null && isFinite(Number(segmentIdx)) ? Math.max(0, parseInt(segmentIdx, 10) || 0) : null;
    if (segIdx != null) {
      const segs = normalizeFlightApronStaySegments(f);
      if (segIdx < segs.length) {
        segs[segIdx].standId = standId || null;
        f.apronStaySegments = segs;
      }
    } else {
      f.standId = standId;
      if (f.token) f.token.apronId = standId;
      f.arrApronId = standId || null;
      f.depApronId = standId || null;
      f.apronStaySegments = [{
        standId: standId || null,
        sibtMin: (f.sibtMin != null && isFinite(f.sibtMin)) ? Number(f.sibtMin) : (f.timeMin != null ? Number(f.timeMin) : 0),
        sobtMin: (f.sobtMin != null && isFinite(f.sobtMin)) ? Number(f.sobtMin) : ((f.timeMin != null ? Number(f.timeMin) : 0) + (f.dwellMin != null ? Number(f.dwellMin) : 0))
      }];
    }
    if (typeof syncFlightApronStayAggregate === 'function') syncFlightApronStayAggregate(f);
    if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
    const touchedSt = [];
    if (prevStandForSched) touchedSt.push(prevStandForSched);
    if (standId) touchedSt.push(standId);
    if (typeof renderFlightList === 'function') {
      renderFlightList(false, false, { scheduleMode: 'incremental', dirtyFlightIds: [f.id], touchedStandIds: touchedSt, skipGanttRefresh: true });
    }
    if (typeof renderFlightGantt === 'function') renderFlightGantt({ skipPathPrep: true });
    if (typeof draw === 'function') {
      // Stand-only change: skip path graph / pro-sim / junction overlays (saves a large 2D pass; geometry unchanged).
      draw({ skipPathGeometryOverlays: true });
    }
    return true;
  }

  function flightScheduleStandWindowMinutes(f) {
    if (!f) return null;
    if (typeof syncFlightApronStayAggregate === 'function') syncFlightApronStayAggregate(f);
    const sibt = (f.sibtMin != null && isFinite(f.sibtMin)) ? Number(f.sibtMin) : (f.timeMin != null ? Number(f.timeMin) : 0);
    const dwell = (f.dwellMin != null && isFinite(f.dwellMin)) ? Number(f.dwellMin) : 0;
    const sobt = (f.sobtMin != null && isFinite(f.sobtMin)) ? Number(f.sobtMin) : (sibt + dwell);
    if (!isFinite(sibt) || !isFinite(sobt) || sobt <= sibt) return null;
    return { sibt, sobt };
  }

  function flightWouldOverlapStandAssignment(f, standId, segmentIdx) {
    if (!f || !standId) return false;
    const segs = typeof normalizeFlightApronStaySegments === 'function' ? normalizeFlightApronStaySegments(f) : [];
    const idx = segmentIdx != null && isFinite(Number(segmentIdx)) ? Math.max(0, parseInt(segmentIdx, 10) || 0) : null;
    const seg = idx != null ? segs[idx] : null;
    const win = seg && isFinite(Number(seg.sibtMin)) && isFinite(Number(seg.sobtMin)) && Number(seg.sobtMin) > Number(seg.sibtMin)
      ? { sibt: Number(seg.sibtMin), sobt: Number(seg.sobtMin) }
      : flightScheduleStandWindowMinutes(f);
    if (!win) return false;
    const target = String(standId);
    const blockedStandIds = new Set([target].concat(duplicateApronStandIdsForStand(target)));
    const flights = state.flights || [];
    for (let i = 0; i < flights.length; i++) {
      const other = flights[i];
      if (!other || other === f || flightBlockedLikeNoWay(other)) continue;
      const segsOther = typeof normalizeFlightApronStaySegments === 'function' ? normalizeFlightApronStaySegments(other) : [];
      if (segsOther.length) {
        for (let j = 0; j < segsOther.length; j++) {
          const os = segsOther[j];
          if (!os || !blockedStandIds.has(String(os.standId || ''))) continue;
          const os0 = Number(os.sibtMin), os1 = Number(os.sobtMin);
          if (isFinite(os0) && isFinite(os1) && win.sibt < os1 && os0 < win.sobt) return true;
        }
      } else if (blockedStandIds.has(String(other.standId || ''))) {
        const ow = flightScheduleStandWindowMinutes(other);
        if (ow && win.sibt < ow.sobt && ow.sibt < win.sobt) return true;
      }
    }
    return false;
  }

  function getCandidatePbbStandsForCode(code, flight) {
    const list = [];
    const allStands = (state.pbbStands || []).concat(state.remoteStands || []);
    allStands.forEach(stand => {
      if (flight && !flightCanUseStand(flight, stand)) return;
      if (!flight && code && getStandCategoryMode(stand) === 'icao') {
        const c = String(code || '').toUpperCase()[0];
        const letters = normalizeAllowedIcaoCategories(stand.allowedIcaoCategories);
        if (letters.length && letters.indexOf(c) < 0) return;
        if (!letters.length && stand.category && String(stand.category).toUpperCase()[0] !== c) return;
      }
      const hasLink = state.apronLinks.some(lk => lk.pbbId === stand.id);
      if (!hasLink) return;
      list.push(stand);
    });
    return list;
  }

  function pickRandom(arr) {
    if (!arr.length) return null;
    const idx = Math.floor(Math.random() * arr.length);
    return arr[idx];
  }

  function resolveStand(flight) {
    const allStands = allStandsForFlightAssignment();
    if (flight.standId) {
      return allStands.find(s => s.id === flight.standId) || null;
    }
    let candidates = getCandidatePbbStandsForCode(flight.code, flight);
    if (!candidates.length) return null;
    const termId = (flight.token && flight.token.terminalId) || null;
    if (termId) {
      const filtered = candidates.filter(st => {
        const allowed = Array.isArray(st.allowedTerminals) ? st.allowedTerminals : null;
        if (allowed && allowed.length) return allowed.includes(termId);
        const t = getTerminalForStand(st);
        return t && t.id === termId;
      });
      if (filtered.length) candidates = filtered;
    }
    const stand = pickRandom(candidates);
    if (stand) flight.standId = stand.id;
    return stand;
  }

  function buildArrivalTimelineFromPts(flight, pts) {
    if (!pts || pts.length < 2) return null;
    const sibtMin = flight.sibtMin != null ? flight.sibtMin : (flight.timeMin != null ? flight.timeMin : 0);
    const baseT = sibtMin * 60;
    const v = Math.max(1, typeof getTaxiwayAvgMoveVelocityForPath === 'function' ? getTaxiwayAvgMoveVelocityForPath(null) : 10);
    const timeline = [];
    let tAcc = baseT;
    timeline.push({ t: tAcc, x: pts[0][0], y: pts[0][1] });
    for (let i = 1; i < pts.length; i++) {
      const [x1,y1] = pts[i-1];
      const [x2,y2] = pts[i];
      const len = Math.hypot(x2-x1, y2-y1);
      const dt = len / v;
      tAcc += dt;
      timeline.push({ t: tAcc, x: x2, y: y2 });
    }
    const sobtMin = flight.sobtMin != null ? flight.sobtMin : (sibtMin + (flight.dwellMin != null ? flight.dwellMin : 0));
    const dwellSec = Math.max(0, (sobtMin - sibtMin) * 60);
    if (dwellSec > 0) {
      tAcc = sobtMin * 60;
      const last = timeline[timeline.length - 1];
      timeline.push({ t: tAcc, x: last.x, y: last.y });
    }
    return timeline;
  }

  function buildDepartureTimelineFromPts(flight, pts) {
    if (!pts || pts.length < 2) return null;
    const sobtMin = flight.sobtMin != null ? flight.sobtMin : (flight.timeMin != null ? flight.timeMin + (flight.dwellMin != null ? flight.dwellMin : 0) : 0);
    const baseT = sobtMin * 60;
    const v = Math.max(1, typeof getTaxiwayAvgMoveVelocityForPath === 'function' ? getTaxiwayAvgMoveVelocityForPath(null) : 10);
    const timeline = [];
    let tAcc = baseT;
    timeline.push({ t: tAcc, x: pts[0][0], y: pts[0][1] });
    for (let i = 1; i < pts.length; i++) {
      const [x1,y1] = pts[i-1];
      const [x2,y2] = pts[i];
      const len = Math.hypot(x2-x1, y2-y1);
      const dt = len / v;
      tAcc += dt;
      timeline.push({ t: tAcc, x: x2, y: y2 });
    }
    return timeline;
  }

  /**
   * Walk distM on the timeline polyline from (fx,fy) on segment segIndex.
   * forward: toward +t; !forward: toward earlier samples (e.g. rear reference from front point).
   */
  function walkTimelinePolylineFromPoint(tl, segIndex, fx, fy, distM, forward) {
    const eps = 1e-6;
    if (!tl || tl.length < 2 || !(distM > eps) || !isFinite(fx) || !isFinite(fy) || !isFinite(distM)) {
      return null;
    }
    if (segIndex < 0 || segIndex > tl.length - 2) return null;
    let rem = distM;
    let x = fx, y = fy;
    let s = segIndex;
    while (rem > eps) {
      if (forward) {
        if (s > tl.length - 2) {
          if (tl.length < 2) return { x, y };
          const n = tl.length, pa = tl[n - 2], pb = tl[n - 1];
          const bx = pb.x - pa.x, by = pb.y - pa.y;
          const bl = Math.hypot(bx, by);
          if (bl < eps) return { x, y };
          const inv = 1 / bl;
          return { x: x + bx * inv * rem, y: y + by * inv * rem };
        }
        const b = tl[s + 1];
        const ddx = b.x - x, ddy = b.y - y;
        const dlen = Math.hypot(ddx, ddy);
        if (dlen < eps) { x = b.x; y = b.y; s++; continue; }
        const step = Math.min(rem, dlen), inv = 1 / dlen;
        x += ddx * inv * step; y += ddy * inv * step; rem -= step;
        if (rem < eps) return { x, y };
        if (dlen - step < eps) { x = b.x; y = b.y; s++; }
      } else {
        if (s < 0) {
          if (tl.length < 2) return { x, y };
          const p0 = tl[0], p1 = tl[1];
          const bx = p0.x - p1.x, by = p0.y - p1.y;
          const bl = Math.hypot(bx, by);
          if (bl < eps) return { x, y };
          const inv = 1 / bl;
          return { x: x + bx * inv * rem, y: y + by * inv * rem };
        }
        const tx = tl[s].x, ty = tl[s].y;
        const ddx = tx - x, ddy = ty - y;
        const dlen = Math.hypot(ddx, ddy);
        if (dlen < eps) { x = tx; y = ty; s--; continue; }
        const step = Math.min(rem, dlen), inv = 1 / dlen;
        x += ddx * inv * step; y += ddy * inv * step; rem -= step;
        if (rem < eps) return { x, y };
        if (dlen - step < eps) { x = tx; y = ty; s--; }
      }
    }
    return { x, y };
  }

  function getFlightPositionAtTime(flight, tSec) {
    const tr = compactPlaybackTrackForFlight(flight);
    const tl = tr ? compactPlaybackTimelineWindow(tr, tSec, 4) : flight.timeline;
    if (!tl || !tl.length) return null;
    if (tSec < tl[0].t || tSec > tl[tl.length - 1].t) return null;
    const i = timelineSegmentIndexAtTime(tl, tSec, false);
    if (i < 0) return null;
    const a = tl[i], b = tl[i+1];
    const span = b.t - a.t || 1;
    const u = (tSec - a.t) / span;
    return {
      x: a.x + (b.x - a.x) * u,
      y: a.y + (b.y - a.y) * u
    };
  }

  function timelineSegmentIndexAtTime(tl, tSec, clampEnd) {
    if (!tl || tl.length < 2) return -1;
    let t = Number(tSec);
    if (!isFinite(t)) return -1;
    const firstT = Number(tl[0].t);
    const lastT = Number(tl[tl.length - 1].t);
    if (!isFinite(firstT) || !isFinite(lastT)) return -1;
    if (t + 1e-9 < firstT) return -1;
    if (t > lastT) {
      if (!clampEnd) return -1;
      t = lastT;
    }
    let lo = 0, hi = tl.length - 1;
    while (lo < hi) {
      const mid = Math.ceil((lo + hi) / 2);
      if (Number(tl[mid].t) <= t + 1e-9) lo = mid;
      else hi = mid - 1;
    }
    let idx = Math.min(lo, tl.length - 2);
    while (idx > 0 && Number(tl[idx].t) > t + 1e-9) idx--;
    while (idx < tl.length - 2 && Number(tl[idx + 1].t) < t - 1e-9) idx++;
    return (t + 1e-9 >= Number(tl[idx].t) && t - 1e-9 <= Number(tl[idx + 1].t)) ? idx : -1;
  }

  function getFlightPoseAtTime(flight, tSec) {
    const tr = compactPlaybackTrackForFlight(flight);
    const tl = tr ? compactPlaybackTimelineWindow(tr, tSec, 80) : flight.timeline;
    if (!tl || !tl.length) return null;
    if (tl.length === 1) {
      const a = tl[0];
      if (tSec + 1e-6 < a.t || tSec - 1e-6 > a.t) return null;
      const dg = a.deadlockGhost === true;
      return { x: a.x, y: a.y, dx: 1, dy: 0, deadlockGhost: dg };
    }
    if (tSec < tl[0].t || tSec > tl[tl.length - 1].t) return null;
    const motionChordEps = 0.08;
    const motionChordEps2 = motionChordEps * motionChordEps;
    function segmentUnitDir(segIdx) {
      if (segIdx < 0 || segIdx > tl.length - 2) return null;
      const p = tl[segIdx], q = tl[segIdx + 1];
      const ddx = q.x - p.x, ddy = q.y - p.y;
      const l2 = ddx * ddx + ddy * ddy;
      if (l2 < motionChordEps2) return null;
      const inv = 1 / Math.sqrt(l2);
      return { dx: ddx * inv, dy: ddy * inv };
    }
    function lastMotionUnitDirBefore(i) {
      for (let j = i - 1; j >= 0; j--) {
        const u = segmentUnitDir(j);
        if (u) return u;
      }
      return null;
    }
    function firstMotionUnitDirFrom(startSeg) {
      for (let j = startSeg; j <= tl.length - 2; j++) {
        const u = segmentUnitDir(j);
        if (u) return u;
      }
      return null;
    }
    function headingForInterval(i) {
      const a = tl[i], b = tl[i + 1];
      const dx = b.x - a.x, dy = b.y - a.y;
      const l2 = dx * dx + dy * dy;
      if (l2 >= motionChordEps2) return { dx: dx, dy: dy };
      const prev = lastMotionUnitDirBefore(i);
      if (prev) return { dx: prev.dx, dy: prev.dy };
      const next = firstMotionUnitDirFrom(i + 1);
      if (next) return { dx: next.dx, dy: next.dy };
      return { dx: 1, dy: 0 };
    }
    function frBicyclePose(R, x, y, lenM, bmin, dg) {
      if (!R || lenM <= 1e-6) return null;
      const vdx = x - R.x, vdy = y - R.y, vl = Math.hypot(vdx, vdy);
      if (vl < bmin) return null;
      return { x, y, dx: vdx / vl, dy: vdy / vl, deadlockGhost: dg };
    }
    const idxAtTime = timelineSegmentIndexAtTime(tl, tSec, false);
    if (idxAtTime >= 0) {
      let i = idxAtTime;
      let a = tl[i], b = tl[i+1];
      let useI = i;
      // At a time-key at the end of [a,b], prefer the outgoing segment so F/R wheels
      // stay consistent with time-forward motion. Last segment has no outgoing.
      if (i + 1 < tl.length - 1) {
        const a2 = tl[i+1], b2 = tl[i+2];
        if (a2 && b2 && b2.t > a2.t && Math.abs(tSec - b.t) < 1e-5) {
          if (Math.abs(b.t - a2.t) < 1e-5) {
            useI = i + 1;
            a = a2;
            b = b2;
          }
        }
      }
      const span = b.t - a.t || 1;
      const u = (tSec - a.t) / span;
      const x = a.x + (b.x - a.x) * u;
      const y = a.y + (b.y - a.y) * u;
      const h = headingForInterval(useI);
      const dg = !!(a.deadlockGhost || b.deadlockGhost);
      const { lenM } = getSimAircraftWorldDimsM(flight);
      const wheelBaseM = 0.55 * lenM;
      const bicycleMin = Math.max(0.15 * motionChordEps, 0.005 * lenM, 0.04);
      let out = frBicyclePose(
        walkTimelinePolylineFromPoint(tl, useI, x, y, wheelBaseM, false), x, y, lenM, bicycleMin, dg);
      if (!out) {
        out = { x, y, dx: h.dx, dy: h.dy, deadlockGhost: dg };
      }
      return out;
    }
    return null;
  }

  /**
   * After EOBT, while on apron_link (departure push/taxi) only: if the bicycle nose points with
   * the ground track step (nose . track &gt; 0), flip dx/dy 180&deg; so the silhouette shows
   * towed/reverse (retro) like R3, without changing (x,y) or the underlying bicycle trace.
   * Does not run before EObT, not off apron, not Arr_taxi, and does not change already-retro
   * pose. Other flights/pathTypes unchanged.
   */
  function applyEobtApronDepTaxiPushbackNoseIfNeeded(flight, tSec, pose) {
    if (!pose || !flight) return pose;
    const m = flight.timeline_meta;
    if (!m || typeof m.eobtSec !== 'number' || !isFinite(m.eobtSec)) return pose;
    if (tSec + 1e-3 < m.eobtSec) return pose;
    const tr = compactPlaybackTrackForFlight(flight);
    const tl = tr ? compactPlaybackTimelineWindow(tr, tSec, 2) : flight.timeline;
    if (!tl || !tl.length) return pose;
    const tKey = Math.round(Number(tSec));
    const byT = Object.create(null);
    for (let i = 0; i < tl.length; i++) {
      const w = tl[i];
      if (!w) continue;
      const tt = Math.round(Number(w.t));
      if (isFinite(tt)) byT[tt] = w;
    }
    const cur = byT[tKey];
    if (!cur) return pose;
    const ph = String(cur.phase || '');
    if (ph !== 'Pushback') return pose;
    const prev = byT[tKey - 1];
    if (!prev) return pose;
    const ddx = cur.x - prev.x, ddy = cur.y - prev.y;
    const dlen = Math.hypot(ddx, ddy);
    if (dlen < 1e-9) return pose;
    const ux = ddx / dlen, uy = ddy / dlen;
    const pl = Math.hypot(pose.dx, pose.dy);
    if (pl < 1e-9) return pose;
    const px = pose.dx / pl, py = pose.dy / pl;
    const dotU = px * ux + py * uy;
    if (dotU <= 0.05) return pose;
    return { x: pose.x, y: pose.y, dx: -pose.dx, dy: -pose.dy, deadlockGhost: !!pose.deadlockGhost };
  }

  /**
   * Pushback tail-first motion: no bicycle model. Fuselage nose=0, tail=100; path
   * sample is station 70 (70% nose→tail). Draw anchor (≈10% aft of nose) at C + h * (0.70−0.1) * lenM
   * with h = unit nose from pose (after applyEobt). Forward taxi leaves pose unchanged.
   */
  function applyApronLinkDepReverseFuselageStation75PoseIfNeeded(flight, tSec, pose) {
    if (!pose || !flight) return pose;
    const m = flight.timeline_meta;
    if (!m || typeof m.eobtSec !== 'number' || !isFinite(m.eobtSec)) return pose;
    const t = Number(tSec);
    if (!isFinite(t) || t + 1e-3 < m.eobtSec) return pose;
    const tr = compactPlaybackTrackForFlight(flight);
    const tl = tr ? compactPlaybackTimelineWindow(tr, tSec, 2) : flight.timeline;
    if (!tl || !tl.length) return pose;
    const tKey = Math.round(t);
    const byT = Object.create(null);
    for (let i = 0; i < tl.length; i++) {
      const w = tl[i];
      if (!w) continue;
      const tt = Math.round(Number(w.t));
      if (isFinite(tt)) byT[tt] = w;
    }
    const cur = byT[tKey];
    if (!cur) return pose;
    const ph = String(cur.phase || '');
    if (ph !== 'Pushback') return pose;
    let a = null;
    let b = null;
    for (let i = 0; i < tl.length - 1; i++) {
      const p = tl[i];
      const q = tl[i + 1];
      if (t + 1e-9 >= p.t && t - 1e-9 <= q.t) {
        a = p;
        b = q;
        break;
      }
    }
    if (!a || !b) return pose;
    const ddx = b.x - a.x;
    const ddy = b.y - a.y;
    const segLen = Math.hypot(ddx, ddy);
    if (segLen < 0.08) return pose;
    const vx = ddx / segLen;
    const vy = ddy / segLen;
    const pl = Math.hypot(pose.dx, pose.dy);
    if (pl < 1e-9) return pose;
    const hx = pose.dx / pl;
    const hy = pose.dy / pl;
    if (hx * vx + hy * vy > -0.05) return pose;
    const C = getFlightPositionAtTime(flight, t);
    if (!C) return pose;
    const { lenM } = getSimAircraftWorldDimsM(flight);
    const NOSE_TO_STATION75_FRAC = 0.70;
    const NOSE_TO_FRONT_WHEEL_FRAC = 0.1;
    const alongNoseM = (NOSE_TO_STATION75_FRAC - NOSE_TO_FRONT_WHEEL_FRAC) * lenM;
    return {
      x: C.x + hx * alongNoseM,
      y: C.y + hy * alongNoseM,
      dx: pose.dx,
      dy: pose.dy,
      deadlockGhost: !!pose.deadlockGhost,
    };
  }

  function getPushbackRearWheelOnPathPoseForDraw(flight, tSec, pose) {
    if (!pose || !flight) return pose;
    const tr = compactPlaybackTrackForFlight(flight);
    const tl = tr ? compactPlaybackTimelineWindow(tr, tSec, 80) : flight.timeline;
    if (!tl || tl.length < 2) return pose;
    const t = Number(tSec);
    if (!isFinite(t)) return pose;
    const tKey = Math.round(t);
    const byT = Object.create(null);
    let transitionStartT = null;
    for (let i = 0; i < tl.length; i++) {
      const w = tl[i];
      if (!w) continue;
      const tt = Math.round(Number(w.t));
      if (isFinite(tt)) byT[tt] = w;
      if (i > 0 && String(w.phase || '') === 'Dep_taxi' && String(tl[i - 1].phase || '') === 'Pushback') {
        transitionStartT = Number(w.t);
      }
    }
    const curPhase = String((byT[tKey] && byT[tKey].phase) || '');
    const prevPhase = String((byT[tKey - 1] && byT[tKey - 1].phase) || '');
    const PUSHBACK_TO_DEP_TAXI_BLEND_SEC = 1.0;
    const inBlend = transitionStartT != null
      && t + 1e-9 >= transitionStartT
      && t <= transitionStartT + PUSHBACK_TO_DEP_TAXI_BLEND_SEC + 1e-9;
    const inPushback = curPhase === 'Pushback' || (curPhase === 'Dep_taxi' && prevPhase === 'Pushback') || inBlend;
    if (!inPushback) return pose;
    let segIdx = -1;
    for (let i = 0; i < tl.length - 1; i++) {
      const a = tl[i], b = tl[i + 1];
      if (t + 1e-9 >= a.t && t - 1e-9 <= b.t) {
        segIdx = i;
        if (i + 1 < tl.length - 1 && Math.abs(t - b.t) < 1e-5) {
          const n = tl[i + 1];
          const nn = tl[i + 2];
          if (n && nn && String(n.phase || '') === 'Pushback' && String(nn.phase || '') === 'Pushback') segIdx = i + 1;
        }
        break;
      }
    }
    if (segIdx < 0) return pose;
    const { lenM } = getSimAircraftWorldDimsM(flight);
    const wheelBaseM = 0.55 * lenM;
    const rear = walkPushbackPolylineFromFront(tl, segIdx, pose.x, pose.y, wheelBaseM);
    if (!rear) return pose;
    const dx = pose.x - rear.x;
    const dy = pose.y - rear.y;
    const dl = Math.hypot(dx, dy);
    if (dl < Math.max(0.005 * lenM, 0.04)) return pose;
    const pushPose = { x: pose.x, y: pose.y, dx: dx / dl, dy: dy / dl, deadlockGhost: !!pose.deadlockGhost };
    if (!inBlend || transitionStartT == null || curPhase === 'Pushback') return pushPose;
    const alpha = Math.max(0, Math.min(1, (t - transitionStartT) / PUSHBACK_TO_DEP_TAXI_BLEND_SEC));
    return blendPoseHeading(pushPose, pose, alpha);
  }

  function blendPoseHeading(fromPose, toPose, alpha) {
    if (!fromPose || !toPose) return fromPose || toPose || null;
    const a = Math.max(0, Math.min(1, Number(alpha) || 0));
    const a0 = Math.atan2(fromPose.dy, fromPose.dx);
    const a1 = Math.atan2(toPose.dy, toPose.dx);
    let da = a1 - a0;
    while (da > Math.PI) da -= Math.PI * 2;
    while (da < -Math.PI) da += Math.PI * 2;
    const th = a0 + da * a;
    return {
      x: toPose.x,
      y: toPose.y,
      dx: Math.cos(th),
      dy: Math.sin(th),
      deadlockGhost: !!(fromPose.deadlockGhost || toPose.deadlockGhost),
    };
  }

  function walkPushbackPolylineFromFront(tl, segIndex, fx, fy, distM) {
    const eps = 1e-6;
    const motionEps = 0.08;
    if (!tl || tl.length < 2 || !(distM > eps)) return null;
    let rem = distM;
    let x = fx, y = fy;
    let s = segIndex;
    let lastUx = null;
    let lastUy = null;
    while (rem > eps && s <= tl.length - 2) {
      const a = tl[s];
      const b = tl[s + 1];
      if (String(a.phase || '') !== 'Pushback' || String(b.phase || '') !== 'Pushback') break;
      const ddx = b.x - x;
      const ddy = b.y - y;
      const dlen = Math.hypot(ddx, ddy);
      if (dlen < eps) {
        const sx = b.x - a.x;
        const sy = b.y - a.y;
        const sl = Math.hypot(sx, sy);
        if (sl > motionEps) {
          lastUx = sx / sl;
          lastUy = sy / sl;
        }
        x = b.x;
        y = b.y;
        s++;
        continue;
      }
      const ux = ddx / dlen;
      const uy = ddy / dlen;
      if (dlen > motionEps) {
        lastUx = ux;
        lastUy = uy;
      }
      const step = Math.min(rem, dlen);
      x += ux * step;
      y += uy * step;
      rem -= step;
      if (rem < eps) return { x, y };
      if (dlen - step < eps) {
        x = b.x;
        y = b.y;
        s++;
      }
    }
    if (lastUx == null || lastUy == null) {
      for (let j = Math.min(segIndex, tl.length - 2); j >= 0; j--) {
        const a = tl[j];
        const b = tl[j + 1];
        if (String(a.phase || '') !== 'Pushback' || String(b.phase || '') !== 'Pushback') continue;
        const sx = b.x - a.x;
        const sy = b.y - a.y;
        const sl = Math.hypot(sx, sy);
        if (sl > motionEps) {
          lastUx = sx / sl;
          lastUy = sy / sl;
          break;
        }
      }
    }
    if (lastUx == null || lastUy == null) return null;
    return { x: x + lastUx * rem, y: y + lastUy * rem };
  }

  function getFlightPoseAtTimeForDraw(flight, tSec) {
    let t = Number(tSec);
    if (!isFinite(t)) return null;
    const trWin = compactPlaybackTrackStartEnd(compactPlaybackTrackForFlight(flight));
    const tl = flight && flight.timeline;
    const t0 = trWin ? trWin.t0 : (tl && tl.length ? tl[0].t : NaN);
    const t1 = trWin ? trWin.t1 : (tl && tl.length ? tl[tl.length - 1].t : NaN);
    if (!isFinite(t0) || !isFinite(t1)) return null;
    if (t + 1e-9 < t0) return null;
    if (t > t1) t = t1;
    return getPushbackRearWheelOnPathPoseForDraw(flight, t, getFlightPoseAtTime(flight, t));
  }
  function simFlightSilhouetteWorldPolygon(f, pose) {
    if (!f || !pose) return [];
    const x = Number(pose.x), y = Number(pose.y), dx = Number(pose.dx), dy = Number(pose.dy);
    if (![x, y, dx, dy].every(isFinite)) return [];
    const len = Math.hypot(dx, dy) || 1;
    const nx = dx / len, ny = dy / len;
    const silN = Number(_acSil.noseX), silWR = Number(_acSil.wingRearX), silUY = Number(_acSil.wingUpperY);
    const silTN = Number(_acSil.tailNeckX), silLY = Number(_acSil.wingLowerY);
    const nX = isFinite(silN) ? silN : 0.6;
    const wRx = isFinite(silWR) ? silWR : -0.5;
    const uY = isFinite(silUY) ? silUY : 0.35;
    const tX = isFinite(silTN) ? silTN : -0.3;
    const lY = isFinite(silLY) ? silLY : -0.35;
    const useDetailSil = _ac2d.useDetailedSilhouette === true;
    const silhouette2D = getApronAircraftDetailedSilhouettePoints();
    const dimsM = getSimAircraftWorldDimsM(f);
    let scaleX, scaleY;
    if (useDetailSil && silhouette2D.length >= 3) {
      const sp = detailedSilhouetteAxisSpans(silhouette2D);
      scaleX = dimsM.lenM / sp.spanX;
      scaleY = dimsM.wingM / sp.spanY;
    } else {
      const xs = [nX, wRx, tX];
      const lenNorm = Math.max(1e-9, Math.max(xs[0], xs[1], xs[2]) - Math.min(xs[0], xs[1], xs[2]));
      const wingNorm = Math.max(1e-9, uY + lY);
      scaleX = dimsM.lenM / lenNorm;
      scaleY = dimsM.wingM / wingNorm;
    }
    const pFwX = nX * scaleX - 0.1 * dimsM.lenM;
    const drawX = x - nx * pFwX;
    const drawY = y - ny * pFwX;
    const pts = (useDetailSil && silhouette2D.length >= 3)
      ? silhouette2D.map(function(p) { return [p[0] * scaleX, p[1] * scaleY]; })
      : [[scaleX * nX, 0], [scaleX * wRx, scaleY * uY], [scaleX * tX, 0], [scaleX * wRx, scaleY * lY]];
    return pts.map(function(p) {
      return [drawX + p[0] * nx - p[1] * ny, drawY + p[0] * ny + p[1] * nx];
    });
  }
  function simFlightPhaseAtTime(f, tSec, pose) {
    if (pose && pose.phase != null) return String(pose.phase || '');
    const seg = typeof flightTimelineSegmentAtSimTime === 'function' ? flightTimelineSegmentAtSimTime(f, tSec) : null;
    return seg && seg.a && seg.a.phase != null ? String(seg.a.phase || '') : '';
  }
  function isFlightParkedAtSimTime(f, tSec) {
    const m = f && f.timeline_meta;
    const t = Number(tSec);
    if (!m || !isFinite(t)) return false;
    const eibtList = Array.isArray(m.eibtSecList) ? m.eibtSecList : (typeof m.eibtSec === 'number' ? [m.eibtSec] : []);
    const eobtList = Array.isArray(m.eobtSecList) ? m.eobtSecList : (typeof m.eobtSec === 'number' ? [m.eobtSec] : []);
    const n = Math.min(eibtList.length, eobtList.length);
    for (let i = 0; i < n; i++) {
      const a = Number(eibtList[i]), b = Number(eobtList[i]);
      if (isFinite(a) && isFinite(b) && t >= a - 1e-3 && t <= b + 1e-3) return true;
    }
    return false;
  }
  function isSecondOrLaterArrTaxiAtTime(f, tSec) {
    const tl = f && Array.isArray(f.timeline) ? f.timeline : null;
    const t = Number(tSec);
    if (!tl || !tl.length || !isFinite(t)) return true;
    let arrTaxiBlockCount = 0;
    let prevArrTaxi = false;
    for (let i = 0; i < tl.length; i++) {
      const ti = Number(tl[i].t);
      if (!isFinite(ti) || ti > t + 1e-9) break;
      const ph = String(tl[i].phase || '').toLowerCase();
      const isArrTaxi = ph.indexOf('arr_taxi') >= 0 || ph.indexOf('arr taxi') >= 0;
      if (isArrTaxi && !prevArrTaxi) arrTaxiBlockCount++;
      prevArrTaxi = isArrTaxi;
    }
    return arrTaxiBlockCount >= 2;
  }
  function flightNeedsTugAtSimTime(f, tSec, pose) {
    if (!f) return false;
    const tr = typeof compactPlaybackTrackForFlight === 'function' ? compactPlaybackTrackForFlight(f) : null;
    if (tr) return compactPlaybackNeedsTugAt(tr, tSec);
    return false;
  }
  function drawFlightTugCar2D(ctx, x, y, nx, ny, lenM, wingM) {
    void lenM;
    void wingM;
    const tugLen = 8;
    const tugWid = 3;
    const cx = x + nx * 3.4;
    const cy = y + ny * 3.4;
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(Math.atan2(ny, nx));
    ctx.fillStyle = '#22c55e';
    ctx.strokeStyle = 'rgba(5,46,22,0.95)';
    ctx.lineWidth = Math.max(0.6, 0.9 / Math.max(state.scale, 0.1));
    ctx.beginPath();
    ctx.rect(-tugLen / 2, -tugWid / 2, tugLen, tugWid);
    ctx.fill();
    ctx.stroke();
    ctx.restore();
  }

  function isFlightPreTouchdownForDraw(f, tSec) {
    if (!PRE_TOUCHDOWN_HALO_ENABLED) return false;
    if (!f || f.arrDep === 'Dep') return false;
    const m = f.timeline_meta;
    if (!m || typeof m.eldtSec !== 'number' || !isFinite(m.eldtSec)) return false;
    const t = Number(tSec);
    if (!isFinite(t)) return false;
    return t < m.eldtSec - 1e-3;
  }

  function isFlightAirsideCycleCompleteAtSimTime(f, tSec) {
    const m = f && f.timeline_meta;
    const t = Number(tSec);
    if (!isFinite(t) || !m || m.error) return false;
    if (typeof m.etotSec !== 'number' || !isFinite(m.etotSec)) return false;
    return t >= m.etotSec - 1e-3;
  }

  
  function isFlightTimelineStationaryAtSimTime(f, tSec) {
    const tr = compactPlaybackTrackForFlight(f);
    const tl = tr ? compactPlaybackTimelineWindow(tr, tSec, 2) : (f && f.timeline);
    if (!tl || tl.length < 2) return false;
    const t = Number(tSec);
    if (!isFinite(t)) return false;
    const t0 = tl[0].t, t1 = tl[tl.length - 1].t;
    if (t < t0 - 1e-9 || t > t1 + 1e-9) return false;
    const stillEps = 0.08;
    for (let i = 0; i < tl.length - 1; i++) {
      const a = tl[i], b = tl[i + 1];
      if (!(t + 1e-9 >= a.t && t - 1e-9 <= b.t)) continue;
      const dt = b.t - a.t;
      if (dt < 1e-9) continue;
      const dist = Math.hypot(b.x - a.x, b.y - a.y);
      if (dist < stillEps) return true;
    }
    return false;
  }

  function isFlightTrailHiddenAtSimTime(f, tSec) {
    if (isFlightAirsideCycleCompleteAtSimTime(f, tSec)) return true;
    if (isFlightTimelineStationaryAtSimTime(f, tSec)) return true;
    return false;
  }

  function getFlightTrailPolylineBackward(f, tEnd, maxDistM) {
    const tr = compactPlaybackTrackForFlight(f);
    const tl = tr ? compactPlaybackTimelineWindow(tr, tEnd, 160) : (f && f.timeline);
    if (!tl || tl.length < 2 || !(maxDistM > 0)) return [];
    const tMin = tl[0].t, tMax = tl[tl.length - 1].t;
    let t = Math.min(Math.max(tEnd, tMin), tMax);
    let seg = Math.max(0, timelineSegmentIndexAtTime(tl, t, true));
    const pts = [];
    function xyAt(T) {
      if (T <= tMin) return [tl[0].x, tl[0].y];
      if (T >= tMax) return [tl[tl.length - 1].x, tl[tl.length - 1].y];
      const i = timelineSegmentIndexAtTime(tl, T, true);
      if (i >= 0) {
        const a = tl[i], b = tl[i + 1];
        const sp = b.t - a.t || 1;
        const uu = (T - a.t) / sp;
        return [a.x + (b.x - a.x) * uu, a.y + (b.y - a.y) * uu];
      }
      return [tl[tl.length - 1].x, tl[tl.length - 1].y];
    }
    pts.push(xyAt(t));
    let rem = maxDistM;
    let curSeg = seg;
    let curT = t;
    let guard = 0;
    while (rem > 1e-6 && curSeg >= 0 && guard++ < 10000) {
      const A = tl[curSeg], B = tl[curSeg + 1];
      const ta = A.t, tb = B.t;
      const dt = tb - ta || 1e-12;
      const distAB = Math.hypot(B.x - A.x, B.y - A.y) || 1e-12;
      let u = Math.max(0, Math.min(1, (curT - ta) / dt));
      if (u < 1e-12) {
        if (curSeg <= 0) break;
        curSeg--;
        curT = tl[curSeg + 1].t;
        continue;
      }
      const distToA = u * distAB;
      if (distToA <= rem) {
        rem -= distToA;
        pts.push([A.x, A.y]);
        curSeg--;
        curT = ta;
      } else {
        const frac = rem / distAB;
        const uu = u - frac;
        const nx = A.x + uu * (B.x - A.x);
        const ny = A.y + uu * (B.y - A.y);
        pts.push([nx, ny]);
        rem = 0;
        break;
      }
    }
    return pts.slice().reverse();
  }

  function getRunwayOptions() {
    const list = [];
    (state.taxiways || []).filter(t => t.pathType === 'runway')
      .forEach(t => list.push({ id: t.id, name: (t.name || '').trim() || 'Runway' }));
    return list;
  }

  function buildRunwayOptionsHtml(selectedId) {
    const opts = [];
    const list = getRunwayOptions();
    if (!list.length) {
      opts.push('<option value=\"\">Runway</option>');
    } else {
      list.forEach(function(o) {
        const sel = selectedId && o.id === selectedId ? ' selected' : '';
        opts.push('<option value=\"' + String(o.id || '').replace(/\"/g, '&quot;') + '\"' + sel + '>' +
          escapeHtml(o.name || o.id || 'Runway') + '</option>');
      });
    }
    return opts.join('');
  }
  function buildTerminalOptionsHtml(selectedId) {
    const opts = [];
    const terms = makeUniqueNamedCopy(state.terminals || [], 'name').map(function(t) {
      return { id: t.id, name: (t.name || '').trim() || 'Building' };
    });
    if (!terms.length) {
      opts.push('<option value=\"\">Building</option>');
    } else {
      if (terms.length > 1) opts.push('<option value=\"\">Random</option>');
      terms.forEach(function(o) {
        const sel = selectedId && o.id === selectedId ? ' selected' : '';
        opts.push('<option value=\"' + String(o.id || '').replace(/\"/g, '&quot;') + '\"' + sel + '>' +
          escapeHtml(o.name || o.id || 'Building') + '</option>');
      });
    }
    return opts.join('');
  }
  function resolveRunwayIdFromInput(raw) {
    const v = (raw || '').trim();
    if (!v) return null;
    const list = getRunwayOptions();
    for (let i = 0; i < list.length; i++) {
      if (list[i].id === v) return v;
    }
    const vl = v.toLowerCase();
    for (let i = 0; i < list.length; i++) {
      if (String(list[i].name || '').trim().toLowerCase() === vl) return list[i].id;
    }
    return undefined;
  }
  function resolveTerminalIdFromInput(raw) {
    const v = (raw || '').trim();
    if (!v) return null;
    const terms = makeUniqueNamedCopy(state.terminals || [], 'name');
    for (let i = 0; i < terms.length; i++) {
      const t = terms[i];
      if (t.id === v) return v;
    }
    const vl = v.toLowerCase();
    for (let i = 0; i < terms.length; i++) {
      const t = terms[i];
      if (String(t.name || '').trim().toLowerCase() === vl) return t.id;
    }
    return undefined;
  }
  function syncFlightAssignInputDisplay(el, f) {
    const role = el.getAttribute('data-role');
    if (role === 'arr') el.value = resolveArrivalRunwayIdForFlight(f) || '';
    else if (role === 'term' || role === 'arrterm') el.value = resolveFlightArrTerminalId(f) || '';
    else if (role === 'depterm') el.value = resolveFlightDepTerminalId(f) || '';
    else if (role === 'dep') el.value = f.depRunwayId || (f.token && f.token.depRunwayId) || '';
    else if (role === 'intdom') el.value = (f && String(f.intDom || '').trim().toLowerCase() === 'dom') ? 'Dom' : 'Int';
  }
  function getRunwayDisplayLabelById(rwId) {
    if (rwId == null || rwId === '') return '—';
    const list = getRunwayOptions();
    const o = list.find(function(x) { return x.id === rwId; });
    return o ? (o.name || o.id || 'Runway') : '—';
  }
  function getTerminalDisplayLabelById(termId) {
    if (termId == null || termId === '') return '—';
    const terms = makeUniqueNamedCopy(state.terminals || [], 'name');
    const t = terms.find(function(x) { return x.id === termId; });
    return t ? ((t.name || '').trim() || 'Building') : '—';
  }
  function resolveFlightBaseTerminalId(f) {
    if (!f) return null;
    return f.terminalId || (f.token && f.token.terminalId) || null;
  }
  function resolveFlightArrTerminalId(f) {
    if (!f) return null;
    return f.arrTerminalId || (f.token && f.token.arrTerminalId) || resolveFlightBaseTerminalId(f);
  }
  function resolveFlightDepTerminalId(f) {
    if (!f) return null;
    return f.depTerminalId || (f.token && f.token.depTerminalId) || resolveFlightBaseTerminalId(f);
  }
  function ensureFlightSplitTerminalDefaults(f) {
    if (!f) return;
    const base = resolveFlightBaseTerminalId(f);
    if (!f.arrTerminalId && base) f.arrTerminalId = base;
    if (!f.depTerminalId && base) f.depTerminalId = base;
    if (f.token) {
      if (!f.token.arrTerminalId && f.arrTerminalId) f.token.arrTerminalId = f.arrTerminalId;
      if (!f.token.depTerminalId && f.depTerminalId) f.token.depTerminalId = f.depTerminalId;
    }
  }
  function flightColorGroupKeyForSim(f, mode) {
    if (mode === 'all') return '*';
    if (mode === 'airline') return 'a:' + (String(f.airlineCode || '').trim() || '—');
    if (mode === 'icao') {
      const c0 = (typeof getCodeForAircraft === 'function') ? String(getCodeForAircraft(f.aircraftType) || 'C').trim().toUpperCase()[0] : 'C';
      return 'i:' + (c0 || 'C');
    }
    if (mode === 'intdom') {
      return 'd:' + ((String(f.intDom || '').trim().toLowerCase() === 'dom') ? 'Dom' : 'Int');
    }
    if (mode === 'building') {
      const arrTid = resolveFlightArrTerminalId(f) || '';
      const depTid = resolveFlightDepTerminalId(f) || '';
      const arrLab = arrTid ? getTerminalDisplayLabelById(arrTid) : '—';
      const depLab = depTid ? getTerminalDisplayLabelById(depTid) : arrLab;
      return 'b:' + arrLab + ' / ' + depLab;
    }
    return '*';
  }
  function buildFlightSim2DColorKeyIndexMap() {
    const mode = state.flightColorMode || 'all';
    if (mode === 'all') return new Map([['*', 0]]);
    const flights = state.flights || [];
    const keys = new Set();
    for (let i = 0; i < flights.length; i++) {
      if (!flights[i]) continue;
      keys.add(flightColorGroupKeyForSim(flights[i], mode));
    }
    const sorted = Array.from(keys).sort();
    const m = new Map();
    for (let j = 0; j < sorted.length; j++) m.set(sorted[j], j);
    return m;
  }
  function resolveFlightSim2DGlyphFillRgba(f, isDeadlockGhost, keyIdxMap, pal, overflow, mode) {
    if (isDeadlockGhost) return 'rgba(148, 163, 184, 0.45)';
    if (mode === 'all') return apron2DGlyphFill();
    const k = flightColorGroupKeyForSim(f, mode);
    const idx = keyIdxMap.get(k);
    if (idx == null || idx >= 10) return overflow;
    return pal[idx] || overflow;
  }
  function parseCssColorToRgbOptional(css) {
    const s = String(css || '').trim();
    const hex6 = s.match(/^#([0-9a-fA-F]{6})$/);
    if (hex6) {
      const h = hex6[1];
      return { r: parseInt(h.slice(0, 2), 16), g: parseInt(h.slice(2, 4), 16), b: parseInt(h.slice(4, 6), 16) };
    }
    const rgba = s.match(/^rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)/);
    if (rgba) return { r: +rgba[1], g: +rgba[2], b: +rgba[3] };
    return null;
  }
  /** Trail stroke gradient: same hue as aircraft fill, fading to transparent along the tail. */
  function simFlightTrailGradientFromFillCss(fillCss) {
    const rgb = parseCssColorToRgbOptional(fillCss);
    if (!rgb) {
      return { near: c2dSimFlightTrailStroke(), far: c2dSimFlightTrailStrokeEnd() };
    }
    return {
      near: 'rgba(' + rgb.r + ',' + rgb.g + ',' + rgb.b + ',0.96)',
      far: 'rgba(' + rgb.r + ',' + rgb.g + ',' + rgb.b + ',0)',
    };
  }
  /** Pre-TD ring: same hue as fill, with soft fill + stroke + glow. */
  function simPreTouchdownHaloFromFillCss(fillCss) {
    const rgb = parseCssColorToRgbOptional(fillCss);
    if (!rgb) {
      return {
        fill: c2dSimPreTouchdownHaloFill(),
        stroke: c2dSimPreTouchdownHaloStroke(),
        shadow: c2dSimPreTouchdownHaloStroke(),
      };
    }
    return {
      fill: 'rgba(' + rgb.r + ',' + rgb.g + ',' + rgb.b + ',0.18)',
      stroke: 'rgba(' + rgb.r + ',' + rgb.g + ',' + rgb.b + ',0.92)',
      shadow: 'rgba(' + rgb.r + ',' + rgb.g + ',' + rgb.b + ',0.55)',
    };
  }
  function syncFlightAssignStripFromFlight(f) {
    const arrEl = document.getElementById('flightAssignStripArr');
    const arrTermEl = document.getElementById('flightAssignStripArrTerm');
    const depTermEl = document.getElementById('flightAssignStripDepTerm');
    const depEl = document.getElementById('flightAssignStripDep');
    const intDomEl = document.getElementById('flightAssignStripIntDom');
    if (f) ensureFlightSplitTerminalDefaults(f);
    if (arrEl) {
      const sid = f ? (resolveArrivalRunwayIdForFlight(f) || '') : '';
      arrEl.innerHTML = buildRunwayOptionsHtml(sid);
      arrEl.value = sid;
    }
    if (intDomEl) {
      intDomEl.value = (f && String(f.intDom || '').trim().toLowerCase() === 'dom') ? 'Dom' : 'Int';
    }
    if (arrTermEl) {
      const tid = f ? (resolveFlightArrTerminalId(f) || '') : '';
      arrTermEl.innerHTML = buildTerminalOptionsHtml(tid);
      arrTermEl.value = tid;
    }
    if (depTermEl) {
      const tid = f ? (resolveFlightDepTerminalId(f) || '') : '';
      depTermEl.innerHTML = buildTerminalOptionsHtml(tid);
      depTermEl.value = tid;
    }
    if (depEl) {
      const did = f ? (f.depRunwayId || (f.token && f.token.depRunwayId) || '') : '';
      depEl.innerHTML = buildRunwayOptionsHtml(did);
      depEl.value = did;
    }
  }
  function syncFlightAssignStrip() {
    const arrEl = document.getElementById('flightAssignStripArr');
    const arrTermEl = document.getElementById('flightAssignStripArrTerm');
    const depTermEl = document.getElementById('flightAssignStripDepTerm');
    const depEl = document.getElementById('flightAssignStripDep');
    const intDomEl = document.getElementById('flightAssignStripIntDom');
    const sel = state.selectedObject;
    const hasFlight = sel && sel.type === 'flight' && sel.id;
    const f = hasFlight ? state.flights.find(function(x) { return x.id === sel.id; }) : null;
    const dis = !f;
    [arrEl, arrTermEl, depTermEl, depEl, intDomEl].forEach(function(el) {
      if (el) el.disabled = dis;
    });
    if (!f) {
      syncFlightAssignStripFromFlight(null);
      return;
    }
    syncFlightAssignStripFromFlight(f);
  }
  function commitFlightAssign(role, flightId, rawValue, st, listEl) {
    const f = st.flights.find(function(x) { return x.id === flightId; });
    if (!f) return;
    const raw = rawValue;
    if (role === 'intdom') {
      const next = (String(raw || '').trim().toLowerCase() === 'dom') ? 'Dom' : 'Int';
      const prev = (String(f.intDom || '').trim().toLowerCase() === 'dom') ? 'Dom' : 'Int';
      if (next === prev) return;
      f.intDom = next;
      syncFlightAssignStripFromFlight(f);
      if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
      if (typeof draw === 'function') draw();
      if (typeof renderFlightList === 'function')
        renderFlightList(false, false, { scheduleMode: 'incremental', dirtyFlightIds: [flightId], touchedStandIds: f.standId ? [f.standId] : [] });
      return;
    }
    var val = null;
    if (role === 'arr' || role === 'dep') {
      const r = resolveRunwayIdFromInput(raw);
      if ((raw || '').trim() && r === undefined) {
        syncFlightAssignStripFromFlight(f);
        return;
      }
      val = r === undefined ? null : r;
    } else if (role === 'term' || role === 'arrterm' || role === 'depterm') {
      const r = resolveTerminalIdFromInput(raw);
      if ((raw || '').trim() && r === undefined) {
        syncFlightAssignStripFromFlight(f);
        return;
      }
      val = r === undefined ? null : r;
    } else return;
    var prevArr = f.arrRunwayId || null;
    var prevDep = f.depRunwayId || (f.token && f.token.depRunwayId) || null;
    var prevArrTerm = resolveFlightArrTerminalId(f) || null;
    var prevDepTerm = resolveFlightDepTerminalId(f) || null;
    if (role === 'arr' && val === prevArr) return;
    if (role === 'dep' && val === prevDep) return;
    if ((role === 'term' || role === 'arrterm') && val === prevArrTerm) return;
    if (role === 'depterm' && val === prevDepTerm) return;
    var prevStand = f.standId || null;
    if (!f.token) f.token = { nodes: ['runway','taxiway','apron','terminal'], runwayId: null, apronId: null, terminalId: null, arrTerminalId: null, depTerminalId: null };
    if (role === 'arr') {
      f.arrRunwayId = val;
      f.token.runwayId = val;
    } else if (role === 'term' || role === 'arrterm') {
      f.arrTerminalId = val;
      f.token.arrTerminalId = val;
      if (!f.depTerminalId) {
        f.depTerminalId = val;
        f.token.depTerminalId = val;
      }
      f.terminalId = val;
      f.token.terminalId = val;
    } else if (role === 'depterm') {
      f.depTerminalId = val;
      f.token.depTerminalId = val;
      if (!f.arrTerminalId) {
        f.arrTerminalId = val;
        f.token.arrTerminalId = val;
      }
      f.terminalId = f.arrTerminalId || val;
      f.token.terminalId = f.terminalId || null;
    } else if (role === 'dep') {
      f.depRunwayId = val;
      f.token.depRunwayId = val;
    }
    syncFlightAssignStripFromFlight(f);
    if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
    var touched = [];
    if (prevStand) touched.push(prevStand);
    if (f.standId) touched.push(f.standId);
    if (typeof renderFlightList === 'function')
      renderFlightList(false, false, { scheduleMode: 'incremental', dirtyFlightIds: [flightId], touchedStandIds: touched });
  }
  function commitFlightAssignField(el, st, listEl) {
    const idVal = el.getAttribute('data-id');
    const role = el.getAttribute('data-role');
    commitFlightAssign(role, idVal, el.value, st, listEl);
  }
  function commitFlightAssignFromStrip(el, st, listEl) {
    const sel = state.selectedObject;
    if (!sel || sel.type !== 'flight' || !sel.id) return;
    const role = el.getAttribute('data-role');
    if (!role) return;
    commitFlightAssign(role, sel.id, el.value, st, listEl);
  }

  /** Flight schedule dynamic AP columns: 10 fixed cells, AP cells, Dep Rw, then S/E groups. */
  const FLIGHT_SCHED_FIXED_BEFORE_AP_COL_COUNT = 10;
  const FLIGHT_SCHED_TRAILING_METRIC_COL_COUNT = 7;
  function flightScheduleLogicalSegmentCount(f) {
    if (!f) return 1;
    const segs = typeof normalizeFlightApronStaySegments === 'function' ? normalizeFlightApronStaySegments(f) : [];
    if (!segs.length) return 1;
    let n = 0;
    let prev = null;
    for (let i = 0; i < segs.length; i++) {
      const sid = segs[i].standId != null ? String(segs[i].standId) : '';
      if (i === 0 || sid !== prev) n++;
      prev = sid;
    }
    return Math.max(1, n);
  }
  function flightScheduleColumnK() {
    const flights = state.flights || [];
    let k = 1;
    for (let i = 0; i < flights.length; i++) k = Math.max(k, flightScheduleLogicalSegmentCount(flights[i]));
    return k;
  }
  function flightSchedColIndex(field, k) {
    const n = Math.max(1, Number(k) || flightScheduleColumnK());
    const apStart = FLIGHT_SCHED_FIXED_BEFORE_AP_COL_COUNT;
    const base = apStart + n + 1;
    if (field === 'ap') return apStart;
    if (field === 'depRunway') return apStart + n;
    if (field === 'sibt') return base;
    if (field === 'sobt') return base + 1;
    if (field === 'eldt') return base + n * 2;
    if (field === 'eibt') return base + n * 2 + 1;
    if (field === 'eobt') return base + n * 2 + 2;
    if (field === 'etot') return base + n * 4 + 1;
    if (field === 'metrics') return base + n * 4 + 2;
    return base;
  }
  function flightScheduleTableColCount(k) {
    return flightSchedColIndex('metrics', k) + FLIGHT_SCHED_TRAILING_METRIC_COL_COUNT + 1;
  }
  /** Backward-compatible aliases for N=1 call sites. Dynamic code should use `flightSchedColIndex`. */
  const FLIGHT_SCHED_TD_SIBT = 12;
  const FLIGHT_SCHED_TD_SOBT = 13;
  const FLIGHT_SCHED_TD_ELDT = 14;
  const FLIGHT_SCHED_TD_EIBT = 15;
  const FLIGHT_SCHED_TD_EOBT = 16;
  const FLIGHT_SCHED_TD_ETOT = 17;
  function ensureFlightAssignStripWired() {
    if (window.__flightAssignStripWired) return;
    const wrap = document.getElementById('flightAssignStrip');
    if (!wrap) return;
    window.__flightAssignStripWired = true;
    wrap.querySelectorAll('.flight-assign-strip-select').forEach(function(inp) {
      inp.addEventListener('change', function(ev) {
        const listEl = document.getElementById('flightList');
        const el = ev.target;
        commitFlightAssignFromStrip(el, state, listEl);
      });
    });
  }

  function _flightListPaintVirtualSlice(listEl) {
    const vs = listEl._flightVirtState;
    if (!vs) return;
    const tbody = listEl.querySelector('.flight-schedule-table[data-virtual-table=\"1\"] tbody');
    if (!tbody) return;
    const flightsSorted = vs.flightsSorted;
    const retStatsAll = vs.retStatsAll;
    const total = flightsSorted.length;
    const rowH = vs.rowH;
    const overscan = vs.overscan;
    const scrollTop = listEl.scrollTop || 0;
    const vh = listEl.clientHeight || 418;
    const start = Math.max(0, Math.floor(scrollTop / rowH) - overscan);
    const rowCount = Math.ceil(vh / rowH) + overscan * 2 + 2;
    const end = Math.min(total, start + rowCount);
    const topPad = start * rowH;
    const botPad = Math.max(0, (total - end) * rowH);
    const parts = [];
    const colCount = flightScheduleTableColCount(vs.apronK || flightScheduleColumnK());
    parts.push('<tr class=\"flight-virt-spacer\" aria-hidden=\"true\" style=\"height:' + topPad + 'px\"><td colspan=\"' + colCount + '\"></td></tr>');
    for (let i = start; i < end; i++) {
      parts.push(_buildFlightListRowHtml(flightsSorted[i], retStatsAll, vs.apronK));
    }
    parts.push('<tr class=\"flight-virt-spacer\" aria-hidden=\"true\" style=\"height:' + botPad + 'px\"><td colspan=\"' + colCount + '\"></td></tr>');
    tbody.innerHTML = parts.join('');
    _flightListWireEvents(listEl, state);
  }
  function _flightListTeardownVirtual(listEl) {
    listEl._flightVirtState = null;
  }
  function _flightListMountVirtual(listEl, flightsSorted, retStatsAll, headerRow, apronK) {
    const prevScroll = listEl.querySelector('.flight-schedule-table[data-virtual-table=\"1\"]') ? (listEl.scrollTop || 0) : 0;
    listEl._flightVirtState = {
      flightsSorted: flightsSorted,
      retStatsAll: retStatsAll,
      rowH: DOM_OPT_FLIGHT_VIRT_ROW_H,
      overscan: DOM_OPT_FLIGHT_VIRT_OVERSCAN,
      apronK: apronK,
      raf: null
    };
    listEl.innerHTML = headerRow + '</tbody></table>';
    const tbl = listEl.querySelector('.flight-schedule-table');
    if (tbl) tbl.setAttribute('data-virtual-table', '1');
    _flightListPaintVirtualSlice(listEl);
    if (prevScroll > 0) listEl.scrollTop = prevScroll;
    if (!listEl._flightVirtScrollBound) {
      listEl._flightVirtScrollBound = true;
      listEl.addEventListener('scroll', function() {
        const vs = listEl._flightVirtState;
        if (!vs || !listEl.querySelector('.flight-schedule-table[data-virtual-table=\"1\"]')) return;
        if (vs.raf) cancelAnimationFrame(vs.raf);
        vs.raf = requestAnimationFrame(function() {
          vs.raf = null;
          _flightListPaintVirtualSlice(listEl);
        });
      });
    }
  }

  function bumpVttArrCacheRev() {
    state.vttArrCacheRev = (state.vttArrCacheRev | 0) + 1;
    bumpRwySepSnapshotStaleGen();
  }
  function getBaseVttArrMinutes(f) {
    if (!f) return 0;
    return 0;
  }
  function getArrRotMinutes(f) {
    if (!f) return 0;
    return 0;
  }
  function getBaseVttDepMinutes(f) {
    if (!f) return 0;
    return 0;
  }
  
  function getBaseVttDepMinutesToLineup(f) {
    if (!f) return 0;
    return 0;
  }
  
  function getDepBlockOutMin(f) {
    const taxi = (typeof getBaseVttDepMinutesToLineup === 'function') ? getBaseVttDepMinutesToLineup(f) : 0;
    const rollBundleSec = (typeof computeDepRollAndLineupOnlySec === 'function')
      ? computeDepRollAndLineupOnlySec(f)
      : (DEP_LINEUP_HOLD_SEC + takeoffRollSecForRunwayTailLenM(0, DEP_TAKEOFF_ACCEL_SMALL_MS2));
    return taxi + rollBundleSec / 60;
  }
  
  function getNormalizedStandDwellBounds(f) {
    let dwell = f.dwellMin != null ? f.dwellMin : 0;
    let minDwell = f.minDwellMin != null ? f.minDwellMin : 0;
    dwell = Math.max(SCHED_DWELL_FLOOR_MIN, dwell);
    minDwell = Math.max(SCHED_DWELL_FLOOR_MIN, minDwell);
    if (minDwell > dwell) minDwell = dwell;
    return { dwell, minDwell };
  }

  /**
   * Apron Gantt SIBT handle: if dwell can shrink (dwell > minDwell), fix SOBT at drag anchor and resize dwell;
   * EIBT shifts by the same Δ as SIBT. If already at min dwell, translate the S block and nudge EOBT/ETOT by Δ.
   */
  function _ganttApplySibtHandleSnappedMinutes(f, mSnapped, dragCtx) {
    if (!f || !dragCtx || flightBlockedLikeNoWay(f)) return false;
    const mClamped = Math.max(0, Number(mSnapped));
    if (!isFinite(mClamped)) return false;
    const anchor = dragCtx.anchorSobt;
    const startS = dragCtx.startSibt;
    const minD = dragCtx.minDwell0;
    const d0 = dragCtx.dwell0;
    if (!(typeof anchor === 'number' && isFinite(anchor)) || !(typeof startS === 'number' && isFinite(startS))) return false;
    const atMinDwell = !(d0 > minD + 1e-9);
    if (atMinDwell) {
      if (typeof applyScheduledGateTimingFromSField === 'function') applyScheduledGateTimingFromSField(f, 'sibt', mClamped);
      const ds = mClamped - startS;
      if (dragCtx.startEobt != null && isFinite(dragCtx.startEobt)) f.eobtMin = dragCtx.startEobt + ds;
      if (dragCtx.startEtot != null && isFinite(dragCtx.startEtot)) f.etotMin = dragCtx.startEtot + ds;
      return true;
    }
    let newDwell = anchor - mClamped;
    let sibtU = mClamped;
    if (newDwell < minD) {
      newDwell = minD;
      sibtU = anchor - minD;
    }
    f.timeMin = sibtU;
    f.sibtMin = sibtU;
    f.sldtMin = scheduledSldtFromSibtMinutes(f, sibtU);
    f.sobtMin = anchor;
    f.dwellMin = Math.max(SCHED_DWELL_FLOOR_MIN, newDwell);
    if (f.minDwellMin != null) {
      f.minDwellMin = Math.max(SCHED_DWELL_FLOOR_MIN, Math.min(f.dwellMin, f.minDwellMin));
    }
    f.stotMin = scheduledStotFromSobtMinutes(f, anchor);
    const deibt = sibtU - startS;
    if (dragCtx.startEibt != null && isFinite(dragCtx.startEibt)) f.eibtMin = dragCtx.startEibt + deibt;
    return true;
  }

  function applyForwardEobtEtotAndDepTaxiDelay(f, eibtMin, etotRunwayCandidateMin) {
    if (!f) return;
    const eibt = eibtMin != null && isFinite(eibtMin) ? eibtMin : 0;
    const block = (typeof getDepBlockOutMin === 'function') ? getDepBlockOutMin(f) : 0;
    const { dwell, minDwell } = getNormalizedStandDwellBounds(f);
    const low = eibt + minDwell;
    const high = eibt + dwell;
    const sobtPref = (f.sobtMin != null)
      ? f.sobtMin
      : (f.sibtMin != null
        ? f.sibtMin + dwell
        : (f.timeMin != null ? f.timeMin + dwell : low));
    const eobt = Math.min(Math.max(sobtPref, low), high);
    const etotDraft = eobt + block;
    let etot = etotDraft;
    if (etotRunwayCandidateMin != null && isFinite(etotRunwayCandidateMin)) {
      etot = Math.max(etotRunwayCandidateMin, etotDraft);
    }
    f.eobtMin = eobt;
    f.etotMin = etot;
    f.depTaxiDelayMin = Math.max(0, etot - etotDraft);
  }

  function pinEarliestEldtToSldtPerRunway(flights) {
    void flights;
  }

  var __schedRetStatsBatchActive = false;
  var __schedRetStatsCached = null;
  var __schedRetExitDistSig = '';
  var __schedRetExitDistMemo = null;
  function scheduleRetExitDistLayoutSig() {
    const tws = state.taxiways || [];
    const parts = [];
    for (let i = 0; i < tws.length; i++) {
      const t = tws[i];
      if (!t || (t.pathType !== 'runway' && t.pathType !== 'runway_exit')) continue;
      let line = String(t.id) + '\x1e' + String(t.pathType) + '\x1e' + JSON.stringify(t.vertices || []);
      if (t.pathType === 'runway' && typeof getTaxiwayDirection === 'function') {
        line += '\x1e' + String(getTaxiwayDirection(t));
      }
      if (t.pathType === 'runway_exit') {
        line += '\x1e' + JSON.stringify(t.allowedRwDirections || []);
        if (typeof getTaxiwayDirection === 'function') {
          line += '\x1e' + String(getTaxiwayDirection(t));
        }
      }
      parts.push(line);
    }
    parts.sort();
    return parts.join('\x1f') + '\x1e' + 'arrivalRetPathEdgeF1V1';
  }
  function bumpScheduleRetExitDistCache() {
    __schedRetExitDistSig = '';
    __schedRetExitDistMemo = null;
  }
  function beginScheduleRetStatsBatch() {
    __schedRetStatsBatchActive = true;
    __schedRetStatsCached = null;
  }
  function endScheduleRetStatsBatch() {
    __schedRetStatsBatchActive = false;
    if (__schedRetStatsCached != null) {
      const sig = scheduleRetExitDistLayoutSig();
      __schedRetExitDistSig = sig;
      __schedRetExitDistMemo = __schedRetStatsCached;
    }
    __schedRetStatsCached = null;
  }
  function getScheduleRetStatsAll() {
    if (__schedRetStatsBatchActive) {
      if (__schedRetStatsCached === null) {
        __schedRetStatsCached = typeof computeRunwayExitDistances === 'function' ? computeRunwayExitDistances() : [];
      }
      return __schedRetStatsCached;
    }
    const sig = scheduleRetExitDistLayoutSig();
    if (sig === __schedRetExitDistSig && __schedRetExitDistMemo && Array.isArray(__schedRetExitDistMemo)) {
      return __schedRetExitDistMemo;
    }
    const res = typeof computeRunwayExitDistances === 'function' ? computeRunwayExitDistances() : [];
    __schedRetExitDistSig = sig;
    __schedRetExitDistMemo = res;
    return res;
  }

  function warmFlightPathsForSchedule(flights) {
    void flights;
  }

  function warmPathsEnsureArrRetRot(flights, forceResampleRet) {
    warmFlightPathsForSchedule(flights);
    return (typeof ensureArrRetRotSampled === 'function')
      ? ensureArrRetRotSampled(flights, !!forceResampleRet)
      : getScheduleRetStatsAll();
  }

  function mutRotCfgEntryForType(configByType, f) {
    const ac = typeof getAircraftInfoByType === 'function' ? getAircraftInfoByType(f.aircraftType) : null;
    const typeKey = f.aircraftType || (ac && ac.id) || (ac && ac.name) || '';
    if (!typeKey) return null;
    if (configByType[typeKey]) return configByType[typeKey];
    const tdMu = (typeof ac?.touchdown_zone_avg_m === 'number') ? ac.touchdown_zone_avg_m : 900;
    const vMu = (typeof ac?.touchdown_speed_avg_ms === 'number') ? ac.touchdown_speed_avg_ms : 70;
    const aMu = (typeof ac?.deceleration_avg_ms2 === 'number') ? ac.deceleration_avg_ms2 : 2.5;
    const tdSigma = Math.round(tdMu * 0.1);
    const vSigma = Math.round(vMu * 0.1);
    const aSigma = Math.round(aMu * 0.1 * 10) / 10;
    configByType[typeKey] = { tdMu, tdSigma, vMu, vSigma, aMu, aSigma };
    return configByType[typeKey];
  }
  /** Same runway resolution as graphPathArrival (token.arrRunwayId before generic runwayId). */
  function resolveArrivalRunwayIdForFlight(f) {
    if (!f) return null;
    const t = f.token || {};
    return t.arrRunwayId || t.runwayId || f.arrRunwayId || null;
  }
  function isValidSampledArrRetForFlight(f, retStatsAll) {
    if (!f || f.sampledArrRet == null) return false;
    if (!Array.isArray(retStatsAll) || !retStatsAll.length) return false;
    const arrRunwayId = resolveArrivalRunwayIdForFlight(f);
    return retStatsAll.some(function(r) {
      if (!r || !r.exit || r.exit.id !== f.sampledArrRet) return false;
      if (arrRunwayId == null) return true;
      return !!(r.runway && r.runway.id === arrRunwayId);
    });
  }
  /** Runway-exit (RET) sampling for Arrival Configuration / schedule RET column. ROT(arr) seconds come from Pro Sim schedule (``ARR_ROT_SEC``), not from this function. */
  function sampleArrRetRotForFlightIfNeeded(f, retStatsAll, configByType, forceResample) {
    if (!f) return;
    const rev = state.vttArrCacheRev | 0;
    if (!forceResample && f.timeline_meta && typeof f.timeline_meta === 'object' &&
        f.timeline_meta.playbackSource === 'des_result') {
      f.__schedRetRotRev = rev;
      return;
    }
    if (!forceResample && f.__schedRetRotRev === rev && isValidSampledArrRetForFlight(f, retStatsAll)) return;
    if (!forceResample && (f.__schedRetRotRev === undefined || f.__schedRetRotRev === null) &&
        f.sampledArrRet != null && f.arrRetFailed === false &&
        isValidSampledArrRetForFlight(f, retStatsAll)) {
      f.__schedRetRotRev = rev;
      return;
    }
    if (f.sampledArrRet != null && !isValidSampledArrRetForFlight(f, retStatsAll)) {
      f.sampledArrRet = null;
      f.arrRetFailed = false;
      f.arrDecelMs2 = null;
    }
    const arrRunwayId = resolveArrivalRunwayIdForFlight(f);
    const cfg = mutRotCfgEntryForType(configByType, f);
    if (!cfg || !retStatsAll || !retStatsAll.length || arrRunwayId == null) {
      f.__schedRetRotRev = rev;
      return;
    }
    const minArrVelRwy = getMinArrVelocityMpsForRunwayId(arrRunwayId);
    const tdSample = sampleNormal(cfg.tdMu, cfg.tdSigma);
    const tdMin = cfg.tdMu * 0.85;
    const tdMax = cfg.tdMu * 1.15;
    const dTd = clamp(tdSample, Math.max(0, tdMin), Math.max(0, tdMax));
    const vSample = sampleNormal(cfg.vMu, cfg.vSigma);
    const vMin = cfg.vMu * 0.85;
    const vMax = cfg.vMu * 1.15;
    const v0 = clamp(vSample, Math.max(0, vMin), Math.max(0, vMax));
    const aSample = sampleNormal(cfg.aMu, cfg.aSigma);
    const aMin = Math.max(0.1, cfg.aMu * 0.85);
    const aMax = Math.min(6,   cfg.aMu * 1.15);
    const aDec = clamp(aSample, aMin, aMax);
    const candidates = retStatsAll.filter(function(r) {
      return !!(r && r.runway && r.runway.id === arrRunwayId && r.exit);
    });
    if (!candidates.length) {
      f.arrDecelMs2 = null;
      f.__schedRetRotRev = rev;
      return;
    }
    let chosen = null;
    candidates.forEach(r => {
      if (chosen) return;
      const distFromTd = Math.max(0, r.distM - dTd);
      const vAt = runwayArrSpeedAndTimeToRet(v0, aDec, distFromTd, minArrVelRwy).vAtRet;
      if (vAt <= r.maxExitVelocity) { chosen = r; }
    });
    if (chosen) {
      f.sampledArrRet = chosen.exit && chosen.exit.id || null;
      f.arrRetFailed = false;
      const MAX_DECEL_MS2 = 15;
      const distFromTdChosen = Math.max(0, chosen.distM - dTd);
      const aDecRot = Math.min(aDec, MAX_DECEL_MS2);
      const rtRunway = runwayArrSpeedAndTimeToRet(v0, aDecRot, distFromTdChosen, minArrVelRwy);
      const vAtChosen = rtRunway.vAtRet;
      const minExitVel = (typeof chosen.minExitVelocity === 'number' && isFinite(chosen.minExitVelocity) && chosen.minExitVelocity > 0)
        ? Math.min(chosen.minExitVelocity, chosen.maxExitVelocity || chosen.minExitVelocity)
        : 15;
      f.arrRunwayIdUsed = arrRunwayId;
      f.arrTdDistM = dTd;
      f.arrRetDistM = chosen.distM;
      f.arrVTdMs = v0;
      f.arrVRetInMs = vAtChosen;
      f.arrVRetOutMs = minExitVel;
      f.arrDecelMs2 = aDecRot;
    } else {
      f.sampledArrRet = null;
      f.arrRetFailed = true;
      f.arrDecelMs2 = null;
    }
    f.__schedRetRotRev = rev;
  }
  function ensureArrRetRotSampled(flights, forceResampleRet) {
    if (!Array.isArray(flights) || !flights.length) return [];
    const configByType = {};
    flights.forEach(f => { mutRotCfgEntryForType(configByType, f); });
    const retStatsAll = getScheduleRetStatsAll();
    flights.forEach(function(f) {
      sampleArrRetRotForFlightIfNeeded(f, retStatsAll, configByType, !!forceResampleRet);
    });
    return retStatsAll;
  }

  function _flightListEmptyHtml(message) {
    return '<div style="font-size:11px;color:#9ca3af;">' + message + '</div>';
  }

  function _renderEmptyFlightListState(listEl, cfgEl) {
    state.flightSchedulePage = 0;
    const pgr = document.getElementById('flightSchedulePager');
    if (pgr) pgr.style.display = 'none';
    _flightListTeardownVirtual(listEl);
    listEl.innerHTML = _flightListEmptyHtml('No flights yet.');
    if (cfgEl) cfgEl.innerHTML = _flightListEmptyHtml('No flights yet.');
    const ganttEl = document.getElementById('allocationGantt');
    if (ganttEl) ganttEl.innerHTML = _flightListEmptyHtml('No flights for Gantt.');
    if (typeof ensureFlightAssignStripWired === 'function') ensureFlightAssignStripWired();
    if (typeof syncFlightAssignStrip === 'function') syncFlightAssignStrip();
  }
  function _updateFlightSchedulePagerUI(totalCount) {
    const pager = document.getElementById('flightSchedulePager');
    if (!pager) return;
    const size = FLIGHT_SCHED_PAGE_SIZE;
    if (!size || size <= 0) {
      pager.style.display = 'none';
      return;
    }
    pager.style.display = 'flex';
    const maxPage = Math.max(0, Math.ceil(totalCount / size) - 1);
    if (state.flightSchedulePage > maxPage) state.flightSchedulePage = maxPage;
    if (state.flightSchedulePage < 0) state.flightSchedulePage = 0;
    const start = state.flightSchedulePage * size;
    const end = Math.min(totalCount, start + size);
    const pageNum = maxPage + 1;
    const cur = state.flightSchedulePage + 1;
    const tEl = document.getElementById('flightSchedulePagerTotal');
    const rEl = document.getElementById('flightSchedulePagerRange');
    if (tEl) tEl.textContent = String(totalCount);
    if (rEl) rEl.textContent = totalCount ? (String(start + 1) + '–' + String(end) + ' · p ' + String(cur) + '/' + String(pageNum)) : '0–0 · p 0/0';
    const bPrev = document.getElementById('btnFlightSchedPrev');
    const bNext = document.getElementById('btnFlightSchedNext');
    if (bPrev) bPrev.disabled = state.flightSchedulePage <= 0;
    if (bNext) bNext.disabled = state.flightSchedulePage >= maxPage;
  }

  /** Same predicate as Arrival Configuration "Failed" row (flight-ui _renderFlightConfigTable failedCounts). */
  function isFlightArrRetFailedInConfigTable(f, retStatsAll) {
    if (!f) return false;
    if (!Array.isArray(retStatsAll) || !retStatsAll.length) return false;
    return f.sampledArrRet === null || typeof f.sampledArrRet === 'undefined';
  }
  function arrivalConfigColumnKeyForFlight(f) {
    if (!f) return '';
    const ac = typeof getAircraftInfoByType === 'function' ? getAircraftInfoByType(f.aircraftType) : null;
    return f.aircraftType || (ac && ac.id) || (ac && ac.name) || '';
  }
  function isFlightCountedInArrivalConfigFailedRow(f, retStats) {
    return isFlightArrRetFailedInConfigTable(f, retStats) && !!arrivalConfigColumnKeyForFlight(f);
  }

  function flightScheduleMinutesForRow(f) {
    const sibt = f.sibtMin != null ? f.sibtMin : (f.timeMin != null ? f.timeMin : 0);
    const dwell = f.dwellMin != null ? f.dwellMin : 0;
    const sobt = f.sobtMin != null ? f.sobtMin : (sibt + dwell);
    return {
      sldt: f.sldtMin != null ? f.sldtMin : Math.max(0, sibt - SCHED_SIBT_MINUS_SLDT_MIN),
      sibt: sibt,
      sobt: sobt,
      stot: f.stotMin != null ? f.stotMin : (sobt + SCHED_STOT_MINUS_SOBT_MIN),
    };
  }

  function flightScheduleProSimTimedCell(displayStr, dotKind) {
    const d = '—';
    const has = displayStr != null && String(displayStr).trim() !== '' && displayStr !== d;
    const text = has ? String(displayStr) : d;
    const muted = has ? '' : ' flight-sched-dot--muted';
    let dotClass = 'flight-sched-dot--green';
    if (dotKind === 'vttarr') dotClass = 'flight-sched-dot--vttarr';
    else if (dotKind === 'dttarr') dotClass = 'flight-sched-dot--dttarr';
    else if (dotKind === 'dttdep') dotClass = 'flight-sched-dot--dttdep';
    else if (dotKind === 'pushback') dotClass = 'flight-sched-dot--pushback';
    else if (dotKind === 'red') dotClass = 'flight-sched-dot--red';
    else if (dotKind === 'pink') dotClass = 'flight-sched-dot--pink';
    return '<span class="flight-sched-cell-inner">' +
      '<span class="flight-sched-dot ' + dotClass + muted + '" aria-hidden="true"></span>' +
      '<span class="flight-sched-cell-text">' + (has ? escapeHtml(text) : d) + '</span></span>';
  }

  function _buildFlightListHeaderHtml(apronK) {
    const k = Math.max(1, Number(apronK) || flightScheduleColumnK());
    const sHeads = [];
    const eHeads = [];
    const apHeads = [];
    for (let i = 1; i <= k; i++) {
      sHeads.push('<th class="flight-col-s' + (i === 1 ? ' flight-col-s-start flight-td-sibt' : '') + '">SIBT' + i + '</th>');
      sHeads.push('<th class="flight-col-s' + (i === k ? ' flight-col-s-last' : '') + '">SOBT' + i + '</th>');
      eHeads.push('<th class="flight-col-e">EIBT' + i + '</th>');
      eHeads.push('<th class="flight-col-e">EOBT' + i + '</th>');
      apHeads.push('<th class="flight-th-mixed">AP' + i + '</th>');
    }
    return '' +
      '<table class="flight-schedule-table">' +
      '<thead><tr>' +
        '<th>Reg</th>' +
        '<th class="flight-th-mixed">Airline</th>' +
        '<th class="flight-th-mixed">Flight Num</th>' +
        '<th>ICAO Code</th>' +
        '<th class="flight-th-mixed">ICAO CAT</th>' +
        '<th>Int/Dom</th>' +
        '<th>Arr Rw</th>' +
        '<th>Arr RET</th>' +
        '<th>Arr Building</th>' +
        '<th>Dep Building</th>' +
        apHeads.join('') +
        '<th>Dep Rw</th>' +
        sHeads.join('') +
        '<th class="flight-col-e flight-col-e-start">ELDT</th>' +
        eHeads.join('') +
        '<th class="flight-col-e">ETOT</th>' +
        '<th class="flight-col-e flight-col-rot flight-th-mixed">ROT(arr)</th>' +
        '<th class="flight-th-mixed">VTT(Arr)</th>' +
        '<th class="flight-th-mixed">DTT(Arr)</th>' +
        '<th class="flight-th-mixed">PUSHBACK</th>' +
        '<th class="flight-th-mixed">DTT(Dep)</th>' +
        '<th class="flight-th-mixed">VTT(Dep)</th>' +
        '<th class="flight-col-e flight-col-rot flight-th-mixed">ROT(dep)</th>' +
        '<th class="flight-td-del"></th>' +
      '</tr></thead>' +
      '<tbody>';
  }

  function flightScheduleSegmentsForDisplay(f, apronK) {
    const raw = typeof normalizeFlightApronStaySegments === 'function' ? normalizeFlightApronStaySegments(f) : [];
    const merged = mergeAdjacentSameStandApronSegments(raw);
    const k = Math.max(1, Number(apronK) || flightScheduleColumnK());
    const out = [];
    for (let i = 0; i < k; i++) out.push(merged[i] || null);
    return out;
  }
  function flightScheduleStandLabelById(standId) {
    if (standId == null || standId === '') return '—';
    const st = typeof findStandById === 'function' ? findStandById(standId) : null;
    if (!st) return String(standId);
    return (st.name && String(st.name).trim()) || String(st.id || standId);
  }
  function _buildFlightListRowHtml(f, retStatsAll, apronK) {
    const k = Math.max(1, Number(apronK) || flightScheduleColumnK());
    const arrRunwayId = resolveArrivalRunwayIdForFlight(f);
    const ac = typeof getAircraftInfoByType === 'function' ? getAircraftInfoByType(f.aircraftType) : null;
    const arrRetFailed = isFlightCountedInArrivalConfigFailedRow(f, retStatsAll);
    let sampledRetName = '—';
    if (arrRetFailed) sampledRetName = 'Failed';
    else if (f.sampledArrRet != null && retStatsAll && retStatsAll.length) {
      const retInfo = retStatsAll.find(r => r.exit && r.exit.id === f.sampledArrRet);
      sampledRetName = retInfo ? (retInfo.name || 'RET') : 'RET';
    }
    const tArrMin = f.sibtMin != null ? f.sibtMin : (f.timeMin != null ? f.timeMin : 0);
    const dwell = f.dwellMin != null ? f.dwellMin : 0;
    const tDepMin = f.sobtMin != null ? f.sobtMin : (tArrMin + dwell);
    const schedDepRotMin = Math.max(0, Number(SCHED_DEP_ROT_MIN) || 2);
    const sldtCalc = (f.sldtMin != null ? f.sldtMin : Math.max(0, tArrMin));
    const stotCalc = (f.stotMin != null) ? f.stotMin : (tDepMin + schedDepRotMin);
    if (f.sibtMin == null || f.sobtMin == null || f.sldtMin == null || f.stotMin == null) {
      f.sldtMin = sldtCalc;
      f.sibtMin = tArrMin;
      f.sobtMin = tDepMin;
      f.stotMin = stotCalc;
    }
    const schedM = flightScheduleMinutesForRow(f);
    const sibtDisp = formatFlightScheduleDateTime(f, schedM.sibt);
    const sobtDisp = formatFlightScheduleDateTime(f, schedM.sobt);
    const segCells = flightScheduleSegmentsForDisplay(f, k);
    function fmtFlightESchedCell(minVal) {
      if (!(typeof minVal === 'number' && isFinite(minVal))) return '—';
      return formatFlightScheduleDateTime(f, minVal);
    }
    const eldtStr = fmtFlightESchedCell(f.eldtMin);
    const eibtStr = fmtFlightESchedCell(f.eibtMin);
    const eobtStr = fmtFlightESchedCell(f.eobtMin);
    const etotStr = fmtFlightESchedCell(f.etotMin);
    const dash = '—';
    const rotArrStr = (f.arrRotSec != null && isFinite(f.arrRotSec)) ? formatSecondsToHHMMSS(f.arrRotSec) : dash;
    const vttArrStr = (f.proSimVttArrSec != null && isFinite(f.proSimVttArrSec)) ? formatSecondsToHHMMSS(f.proSimVttArrSec) : dash;
    const dttArrStr = (f.proSimDttArrSec != null && isFinite(f.proSimDttArrSec)) ? formatSecondsToHHMMSS(f.proSimDttArrSec) : dash;
    const pushbackStr = (f.proSimPushbackSec != null && isFinite(f.proSimPushbackSec)) ? formatSecondsToHHMMSS(f.proSimPushbackSec) : dash;
    const dttDepStr = (f.proSimDttDepSec != null && isFinite(f.proSimDttDepSec)) ? formatSecondsToHHMMSS(f.proSimDttDepSec) : dash;
    const vttDepStr = (f.proSimVttDepSec != null && isFinite(f.proSimVttDepSec)) ? formatSecondsToHHMMSS(f.proSimVttDepSec) : dash;
    const rotDepStr = (f.proSimDepLineupSec != null && isFinite(f.proSimDepLineupSec)) ? formatSecondsToHHMMSS(f.proSimDepLineupSec) : dash;
    const rotArrCell = flightScheduleProSimTimedCell(rotArrStr, 'green');
    const vttArrCell = flightScheduleProSimTimedCell(vttArrStr, 'vttarr');
    const dttArrCell = flightScheduleProSimTimedCell(dttArrStr, 'dttarr');
    const pushbackCell = flightScheduleProSimTimedCell(pushbackStr, 'pushback');
    const dttDepCell = flightScheduleProSimTimedCell(dttDepStr, 'dttdep');
    const vttDepCell = flightScheduleProSimTimedCell(vttDepStr, 'red');
    const rotDepCell = flightScheduleProSimTimedCell(rotDepStr, 'pink');
    const depRunwayId = f.depRunwayId || (f.token && f.token.depRunwayId);
    ensureFlightSplitTerminalDefaults(f);
    const arrTermId = resolveFlightArrTerminalId(f);
    const depTermId = resolveFlightDepTerminalId(f);
    const arrRwRead = escapeHtml(getRunwayDisplayLabelById(arrRunwayId));
    const arrBuildingRead = escapeHtml(getTerminalDisplayLabelById(arrTermId));
    const depBuildingRead = escapeHtml(getTerminalDisplayLabelById(depTermId));
    const depRwRead = escapeHtml(getRunwayDisplayLabelById(depRunwayId));
    function segTimeCell(seg, key, cls) {
      if (!seg) return '<td class="flight-td-time ' + cls + '" data-empty="1">—</td>';
      const m = Number(seg[key]);
      const txt = isFinite(m) ? formatFlightScheduleDateTime(f, m) : '—';
      return '<td class="flight-td-time ' + cls + '" data-sched-min="' + (isFinite(m) ? m : '') + '">' + escapeHtml(txt) + '</td>';
    }
    function eSeriesCell(minVal, labelIdx) {
      const txt = fmtFlightESchedCell(minVal);
      return '<td class="flight-td-time flight-col-e" data-e-series-index="' + labelIdx + '">' + escapeHtml(txt) + '</td>';
    }
    const sCells = segCells.map(function(seg, idx) {
      return [
        segTimeCell(seg, 'sibtMin', 'flight-col-s' + (idx === 0 ? ' flight-col-s-start flight-td-sibt' : '')),
        segTimeCell(seg, 'sobtMin', 'flight-col-s' + (idx === k - 1 ? ' flight-col-s-last' : ''))
      ].join('');
    }).join('');
    const eCells = segCells.map(function(_seg, idx) {
      const eibtList = flightEMinListForSchedule(f, 'eibtMinList', 'eibtSecList', 'eibtMin');
      const eobtList = flightEMinListForSchedule(f, 'eobtMinList', 'eobtSecList', 'eobtMin');
      return [
        eSeriesCell(eibtList[idx] != null ? eibtList[idx] : null, idx + 1),
        eSeriesCell(eobtList[idx] != null ? eobtList[idx] : null, idx + 1)
      ].join('');
    }).join('');
    const apCells = segCells.map(function(seg) {
      const lab = seg ? flightScheduleStandLabelById(seg.standId) : '—';
      return '<td class="flight-td-readonly" data-empty="' + (seg ? '0' : '1') + '">' + escapeHtml(lab) + '</td>';
    }).join('');
    const aircraftTypeLabel = ac ? (ac.name || ac.id || '') : (f.aircraftType || '—');
    const codeIcao = (ac && ac.icao) ? ac.icao : (f.code || '—');
    const intDomVal = (String(f.intDom || '').trim().toLowerCase() === 'dom') ? 'Dom' : 'Int';
    return '' +
      '<tr class="flight-data-row obj-item" data-id="' + f.id + '">' +
        '<td class="flight-td-reg">' + escapeHtml(f.reg || '') + '</td>' +
        '<td class="flight-td-reg">' + escapeHtml(f.airlineCode || '') + '</td>' +
        '<td class="flight-td-reg">' + escapeHtml(f.flightNumber || '') + '</td>' +
        '<td>' + escapeHtml(aircraftTypeLabel) + '</td>' +
        '<td>' + escapeHtml(codeIcao) + '</td>' +
        '<td class="flight-td-readonly" title="Edit in Int/Dom above when flight is selected">' + escapeHtml(intDomVal) + '</td>' +
        '<td class="flight-td-readonly">' + arrRwRead + '</td>' +
        '<td class="flight-td-arr-ret' + (arrRetFailed ? ' flight-td-arr-ret-failed' : '') + '">' + (arrRetFailed ? 'Failed' : escapeHtml(sampledRetName)) + '</td>' +
        '<td class="flight-td-readonly">' + arrBuildingRead + '</td>' +
        '<td class="flight-td-readonly">' + depBuildingRead + '</td>' +
        apCells +
        '<td class="flight-td-readonly">' + depRwRead + '</td>' +
        sCells +
        '<td class="flight-td-time flight-col-e flight-col-e-start">' + escapeHtml(eldtStr) + '</td>' +
        eCells +
        '<td class="flight-td-time flight-col-e">' + escapeHtml(etotStr) + '</td>' +
        '<td class="flight-td-time flight-col-e flight-col-rot">' + rotArrCell + '</td>' +
        '<td class="flight-td-time">' + vttArrCell + '</td>' +
        '<td class="flight-td-time">' + dttArrCell + '</td>' +
        '<td class="flight-td-time">' + pushbackCell + '</td>' +
        '<td class="flight-td-time">' + dttDepCell + '</td>' +
        '<td class="flight-td-time">' + vttDepCell + '</td>' +
        '<td class="flight-td-time flight-col-e flight-col-rot">' + rotDepCell + '</td>' +
        '<td class="flight-td-del"><button type="button" class="obj-item-delete" data-del="' + f.id + '">×</button></td>' +
      '</tr>';
  }

  function _buildFlightListRowsHtml(flightsSorted, retStatsAll, apronK) {
    return flightsSorted.map(function(f) {
      return _buildFlightListRowHtml(f, retStatsAll, apronK);
    });
  }

  const FLIGHT_LIST_PATH_YIELD_CHUNK = 6;
  const FLIGHT_LIST_ASYNC_PATH_MIN = 8;
  function _renderFlightListDomAndSchedule(flightsSorted, schedFull, dirtySet, standSet, listEl, cfgEl, retStatsAll, domOpt) {
    const skipGanttRefresh = domOpt && domOpt.skipGanttRefresh;
    const apronK = flightScheduleColumnK();
    const headerRow = _buildFlightListHeaderHtml(apronK);
    const dirtyIds = [];
    dirtySet.forEach(function(id) { if (id != null && id !== '') dirtyIds.push(id); });
    const deferOnlyDirty = false;
    if (schedFull) {
      if (typeof computeScheduledDisplayTimes === 'function') computeScheduledDisplayTimes(state.flights);
    } else {
      if (!deferOnlyDirty && typeof computeScheduledDisplayTimesIncremental === 'function')
        computeScheduledDisplayTimesIncremental(state.flights, dirtySet, standSet);
    }
    flightsSorted.sort((a, b) => (a.sibtMin != null ? a.sibtMin : (a.timeMin != null ? a.timeMin : 0)) - (b.sibtMin != null ? b.sibtMin : (b.timeMin != null ? b.timeMin : 0)));
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
      _flightListMountVirtual(listEl, flightsSorted, retStatsAll, headerRow, apronK);
    } else {
      _flightListTeardownVirtual(listEl);
      const dataRows = _buildFlightListRowsHtml(flightsForDom, retStatsAll, apronK);
      listEl.innerHTML = headerRow + dataRows.join('') + '</tbody></table>';
      const tbl0 = listEl.querySelector('.flight-schedule-table');
      if (tbl0) {
        if (usePagination) tbl0.setAttribute('data-virtual-table', '1');
        else tbl0.removeAttribute('data-virtual-table');
      }
      _flightListWireEvents(listEl, state);
    }
    _renderFlightConfigTable(cfgEl, flightsSorted);
    if (typeof ensureFlightAssignStripWired === 'function') ensureFlightAssignStripWired();
    if (typeof syncFlightAssignStrip === 'function') syncFlightAssignStrip();
    if (!skipGanttRefresh && typeof renderFlightGantt === 'function') renderFlightGantt({ skipPathPrep: true });
  }
  function _renderFlightListAfterPathEnsure(flightsSorted, schedFull, forceResampleRet, dirtySet, standSet, listEl, cfgEl, scheduleOpts) {
    if (forceResampleRet && typeof bumpVttArrCacheRev === 'function') bumpVttArrCacheRev();
    let retStatsAll = [];
    if (schedFull) {
      retStatsAll = (typeof ensureArrRetRotSampled === 'function')
        ? ensureArrRetRotSampled(flightsSorted, !!forceResampleRet)
        : (typeof getScheduleRetStatsAll === 'function' ? getScheduleRetStatsAll() : ((typeof computeRunwayExitDistances === 'function') ? computeRunwayExitDistances() : []));
    } else {
      const dirtyFlights = flightsSorted.filter(function(f) { return dirtySet.has(f.id); });
      const dirtyForRet = dirtyFlights.filter(function(f) { return f; });
      if (dirtyForRet.length && typeof ensureArrRetRotSampled === 'function')
        retStatsAll = ensureArrRetRotSampled(dirtyForRet, false);
      else
        retStatsAll = (typeof getScheduleRetStatsAll === 'function') ? getScheduleRetStatsAll() : ((typeof computeRunwayExitDistances === 'function') ? computeRunwayExitDistances() : []);
    }
    const domOpt = (scheduleOpts && scheduleOpts.skipGanttRefresh) ? { skipGanttRefresh: true } : null;
    _renderFlightListDomAndSchedule(flightsSorted, schedFull, dirtySet, standSet, listEl, cfgEl, retStatsAll, domOpt);
  }

  function renderFlightList(skipAutoAllocate, forceResampleRet, scheduleOpts, onDone) {
    const listEl = document.getElementById('flightList');
    const cfgEl = document.getElementById('flightConfigList');
    const cb = typeof onDone === 'function' ? onDone : null;
    if (!listEl) return;
    if (!state.flights.length) {
      _renderEmptyFlightListState(listEl, cfgEl);
      if (typeof syncProSimButtonFromDesignerPageState === 'function') syncProSimButtonFromDesignerPageState();
      if (cb) cb();
      return;
    }
    if (scheduleOpts && scheduleOpts.pageTurnOnly === true && FLIGHT_SCHED_PAGE_SIZE > 0) {
      const flightsSorted = state.flights.slice();
      flightsSorted.sort((a, b) => (a.sibtMin != null ? a.sibtMin : (a.timeMin != null ? a.timeMin : 0)) - (b.sibtMin != null ? b.sibtMin : (b.timeMin != null ? b.timeMin : 0)));
      beginScheduleRetStatsBatch();
      var retStatsAll2 = [];
      try {
        retStatsAll2 = (typeof getScheduleRetStatsAll === 'function')
          ? getScheduleRetStatsAll()
          : ((typeof computeRunwayExitDistances === 'function') ? computeRunwayExitDistances() : []);
        _renderFlightListDomAndSchedule(flightsSorted, false, new Set(), new Set(), listEl, cfgEl, retStatsAll2, { skipGanttRefresh: true });
      } finally {
        endScheduleRetStatsBatch();
      }
      if (typeof syncProSimButtonFromDesignerPageState === 'function') syncProSimButtonFromDesignerPageState();
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
    flightsSorted.sort((a, b) => (a.sibtMin != null ? a.sibtMin : (a.timeMin != null ? a.timeMin : 0)) - (b.sibtMin != null ? b.sibtMin : (b.timeMin != null ? b.timeMin : 0)));
    function runTail() {
      beginScheduleRetStatsBatch();
