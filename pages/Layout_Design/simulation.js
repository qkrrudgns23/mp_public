    if (hEl) hEl.value = String(nh);
    ensurePbbBoardingWallGeometry(pbb);
    const arm = Number(pbb.pbbArmLenM);
    if (isFinite(arm) && arm > 0) applyPbbArmLengthToBridgeEnds(pbb, arm);
    bumpPathPolylineCacheRev();
  }
  const pbbBoardingWidthEl = document.getElementById('pbbBoardingWidth');
  if (pbbBoardingWidthEl) {
    pbbBoardingWidthEl.addEventListener('change', function() {
      if (state.selectedObject && state.selectedObject.type === 'pbb') {
        applyPbbBoardingAreaDimsFromInputs(state.selectedObject.obj);
        updateObjectInfo();
        renderObjectList();
        draw();
        update3DSceneWhenVisible();
      }
    });
  }
  const pbbBoardingHeightEl = document.getElementById('pbbBoardingHeight');
  if (pbbBoardingHeightEl) {
    pbbBoardingHeightEl.addEventListener('change', function() {
      if (state.selectedObject && state.selectedObject.type === 'pbb') {
        applyPbbBoardingAreaDimsFromInputs(state.selectedObject.obj);
        updateObjectInfo();
        renderObjectList();
        draw();
        update3DSceneWhenVisible();
      }
    });
  }
  const standAircraftAccessEl = document.getElementById('standAircraftAccess');
  if (standAircraftAccessEl) {
    standAircraftAccessEl.addEventListener('change', function(ev) {
      const target = ev.target;
      if (!target || !target.classList.contains('aircraft-type-check')) return;
      syncChoiceChipStates(standAircraftAccessEl);
      if (!state.selectedObject || state.selectedObject.type !== 'pbb') return;
      const pbbAc = state.selectedObject.obj;
      applyUnifiedStandConstraintFromPanelToObject(pbbAc, 'standIcaoCategories', 'standAircraftAccess');
      renderAircraftConstraintChoices('standAircraftAccess', pbbAc.allowedAircraftTypes, pbbAc.allowedIcaoCategories);
      updateObjectInfo();
      renderObjectList();
      draw();
    });
  }

  const remoteNameInput = document.getElementById('remoteName');
  if (remoteNameInput) {
    remoteNameInput.addEventListener('change', function() {
      if (state.selectedObject && state.selectedObject.type === 'remote') {
        const st = state.selectedObject.obj;
        const raw = (this.value || '').trim();
        if (raw && findDuplicateLayoutName('remote', st.id, raw)) {
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
      const w = getFlightAirsideWindowSec(f);
      if (!w) return;
      if (w.t0 < minT) minT = w.t0;
      if (w.t1 > maxT) maxT = w.t1;
    });
    if (!isFinite(minT) || !isFinite(maxT)) {
      minT = 0;
      maxT = 0;
    }
    let simLo = minT;
    if (PLAYBACK_LEAD_BEFORE_FIRST_TD_SEC > 0) {
      const firstTdS = minFirstArrivalTouchdownSecAmongFlights();
      if (firstTdS != null) {
        simLo = Math.max(0, firstTdS - PLAYBACK_LEAD_BEFORE_FIRST_TD_SEC);
      }
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
      const dwellMin = (f.sobtMin_d != null && f.sibtMin_d != null) ? (f.sobtMin_d - f.sibtMin_d) : (f.dwellMin || 0);
      const dwellSec = Math.max(0, dwellMin * 60);
      const start = Math.max(0, end - dwellSec);
      if (end > start) intervals.push({ start, end });
    });
    intervals.sort((a, b) => a.start - b.start);
    return intervals;
  }

  function isStandOccupiedAtSimSec(standId, tSec) {
    if (!standId || !state.hasSimulationResult) return false;
    const t = Number(tSec);
    if (!isFinite(t)) return false;
    const flights = state.flights || [];
    for (let i = 0; i < flights.length; i++) {
      const f = flights[i];
      if (!f || f.standId !== standId) continue;
      const m = f.timeline_meta;
      if (m && typeof m.eibtSec === 'number' && typeof m.eobtSec === 'number') {
        if (t + 1e-3 >= m.eibtSec && t <= m.eobtSec + 1e-3) return true;
        continue;
      }
      if (f.arrDep !== 'Dep' && (f.noWayArr || f.arrRetFailed)) {
        const eldtMin = flightEMinutesPrefer(f, ['eldtMin'], flightEMinutesPrefer(f, ['timeMin'], 0));
        const eibtMin = flightEMinutesPrefer(f, ['eibtMin'], eldtMin + 15);
        const eobtMin = flightEMinutesPrefer(f, ['eobtMin'], eibtMin + (typeof f.dwellMin === 'number' && isFinite(f.dwellMin) ? f.dwellMin : 45));
        const eibtS = eibtMin * 60;
        const eobtS = eobtMin * 60;
        if (t + 1e-3 >= eibtS && t <= eobtS + 1e-3) return true;
      }
    }
    return false;
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

  function flightCanUseStand(f, stand) {
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
    const ft = (f.terminalId || (f.token && f.token.terminalId)) || null;
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

  function assignStandToFlight(f, standId) {
    if (!f) return false;
    if (standId) {
      const allStands = allStandsForFlightAssignment();
      const stand = allStands.find(function(s) { return s.id === standId; });
      if (!flightCanUseStand(f, stand)) {
        alert("Stand constraints or selected building do not match this aircraft, so it cannot be assigned.");
        return false;
      }
    }
    const prevStandForSched = f.standId || null;
    f.standId = standId;
    if (f.token) f.token.apronId = standId;
    delete f.sobtMin_orig;
    delete f.sldtMin_orig;
    delete f.sibtMin_orig;
    delete f.stotMin_orig;
    delete f.eldtMin_orig;
    delete f.eibtMin_orig;
    delete f.eobtMin_orig;
    delete f.etotMin_orig;
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
    const sibtMin_d = flight.sibtMin_d != null ? flight.sibtMin_d : (flight.timeMin != null ? flight.timeMin : 0);
    const baseT = sibtMin_d * 60;
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
    const sobtMin_d = flight.sobtMin_d != null ? flight.sobtMin_d : (sibtMin_d + (flight.dwellMin != null ? flight.dwellMin : 0));
    const dwellSec = Math.max(0, (sobtMin_d - sibtMin_d) * 60);
    if (dwellSec > 0) {
      tAcc = sobtMin_d * 60;
      const last = timeline[timeline.length - 1];
      timeline.push({ t: tAcc, x: last.x, y: last.y });
    }
    return timeline;
  }

  function buildDepartureTimelineFromPts(flight, pts) {
    if (!pts || pts.length < 2) return null;
    const sobtMin_d = flight.sobtMin_d != null ? flight.sobtMin_d : (flight.timeMin != null ? flight.timeMin + (flight.dwellMin != null ? flight.dwellMin : 0) : 0);
    const baseT = sobtMin_d * 60;
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
    const tl = flight.timeline;
    if (!tl || !tl.length) return null;
    if (tSec < tl[0].t || tSec > tl[tl.length - 1].t) return null;
    for (let i = 0; i < tl.length - 1; i++) {
      const a = tl[i], b = tl[i+1];
      if (tSec >= a.t && tSec <= b.t) {
        const span = b.t - a.t || 1;
        const u = (tSec - a.t) / span;
        return {
          x: a.x + (b.x - a.x) * u,
          y: a.y + (b.y - a.y) * u
        };
      }
    }
    return null;
  }

  function getFlightPoseAtTime(flight, tSec) {
    const tl = flight.timeline;
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
    for (let i = 0; i < tl.length - 1; i++) {
      let a = tl[i], b = tl[i+1];
      if (tSec >= a.t && tSec <= b.t) {
        let useI = i;
        // At a time-key at the end of [a,b], the first matching segment is the *incoming* leg.
        // The bicycle (rear on polyline) + that chord can show nose 180° off next-second motion.
        // Prefer the *outgoing* segment [b, next] (same (x,y) at t=b.t, u=0) so F/R wheels stay
        // consistent with time-forward motion. Last segment has no outgoing — keep i.
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
    const tl = flight.timeline;
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
    const tl = flight.timeline;
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
    const tl = flight.timeline;
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

