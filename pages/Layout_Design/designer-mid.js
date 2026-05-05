      pushUndo();
      state.layoutImageOverlay = null;
      layoutImageBitmap = null;
      layoutImageBitmapSrc = '';
      if (gridLayoutImageFileEl) gridLayoutImageFileEl.value = '';
      syncPanelFromState();
      draw();
    });
  }
  commitGridLayoutImageNumericChange('gridLayoutImageOpacity', function(input) {
    state.layoutImageOverlay.opacity = clampLayoutImageOpacity(input.value);
  });
  commitGridLayoutImageNumericChange('gridLayoutImageWidthM', function(input) {
    applyLayoutImageWidthByAspect(input.value);
  });
  commitGridLayoutImageNumericChange('gridLayoutImageHeightM', function(input) {
    applyLayoutImageHeightByAspect(input.value);
  });
  commitGridLayoutImageNumericChange('gridLayoutImageCol', function(input) {
    state.layoutImageOverlay.topLeftCol = clampLayoutImagePoint(input.value, state.layoutImageOverlay.topLeftCol);
  });
  commitGridLayoutImageNumericChange('gridLayoutImageRow', function(input) {
    state.layoutImageOverlay.topLeftRow = clampLayoutImagePoint(input.value, state.layoutImageOverlay.topLeftRow);
  });

  document.getElementById('terminalName').addEventListener('change', function() {
    const t = getCurrentTerminal();
    if (t) {
      const raw = (this.value || '').trim();
      if (raw && findDuplicateLayoutName('terminal', t.id, raw)) {
        alertDuplicateLayoutName();
        this.value = t.name || '';
        return;
      }
      t.name = raw || t.name;
      draw();
      updateObjectInfo();
      if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
    }
  });
  const buildingTypeInput = document.getElementById('buildingType');
  if (buildingTypeInput) {
    buildingTypeInput.addEventListener('change', function() {
      const nextType = normalizeBuildingType(this.value || BUILDING_TYPE_DEFAULT);
      const t = getCurrentTerminal();
      const nameInput = document.getElementById('terminalName');


      const nextDefaultName = getDefaultBuildingNameForType(nextType, t ? t.id : null);
      if (t) {
        t.buildingType = nextType;
        if (findDuplicateLayoutName('terminal', t.id, nextDefaultName)) {
          alertDuplicateLayoutName();
          if (nameInput) nameInput.value = t.name || '';
        } else {
          t.name = nextDefaultName;
          if (nameInput) nameInput.value = nextDefaultName;
        }
      } else if (nameInput) {
        nameInput.value = nextDefaultName;
      }
      updateObjectInfo();
      renderObjectList();
      draw();
      update3DSceneWhenVisible();
      if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
    });
  }
  function recomputeTerminalFloorHeight() {
    const t = getCurrentTerminal();
    if (!t) return;
    const floorsInput = document.getElementById('terminalFloors');
    const f2fInput = document.getElementById('terminalFloorToFloor');
    const totalInput = document.getElementById('terminalFloorHeight');
    let floors = floorsInput ? parseInt(floorsInput.value, 10) : t.floors;
    let f2f = f2fInput ? Number(f2fInput.value) : t.floorToFloor;
    floors = Math.max(1, floors || 1);
    f2f = Math.max(0.5, f2f || 4);
    const totalH = floors * f2f;
    t.floors = floors;
    t.floorToFloor = f2f;
    t.floorHeight = totalH;
    if (floorsInput) floorsInput.value = floors;
    if (f2fInput) f2fInput.value = f2f;
    if (totalInput) totalInput.value = totalH;
    draw();
    updateObjectInfo();
    update3DSceneWhenVisible();
  }
  document.getElementById('terminalFloors').addEventListener('change', recomputeTerminalFloorHeight);
  document.getElementById('terminalFloorToFloor').addEventListener('change', recomputeTerminalFloorHeight);
  document.getElementById('terminalDepartureCapacity').addEventListener('change', function() {
    const t = getCurrentTerminal();
    if (t) { t.departureCapacity = Math.max(0, parseInt(this.value, 10) || 0); updateObjectInfo(); }
  });
  document.getElementById('terminalArrivalCapacity').addEventListener('change', function() {
    const t = getCurrentTerminal();
    if (t) { t.arrivalCapacity = Math.max(0, parseInt(this.value, 10) || 0); updateObjectInfo(); }
  });

  document.getElementById('standName').addEventListener('change', function() {
    if (state.selectedObject && state.selectedObject.type === 'pbb') {
      const pbb = state.selectedObject.obj;
      const raw = (this.value || '').trim();
      if (raw && findDuplicateLayoutName('pbb', pbb.id, raw)) {
        alertDuplicateLayoutName();
        this.value = pbb.name || '';
        return;
      }
      pbb.name = raw;
      updateObjectInfo();
      renderObjectList();
      draw();
    }
  });
  const standIcaoCategoriesHost = document.getElementById('standIcaoCategories');
  if (standIcaoCategoriesHost) {
    standIcaoCategoriesHost.addEventListener('change', function(ev) {
      const t = ev.target;
      if (!t || !t.classList.contains('icao-letter-check')) return;
      let letters = readIcaoCategoriesFromHost('standIcaoCategories');
      if (!letters.length) {
        letters = ['C'];
        applyIcaoCategoriesToHost('standIcaoCategories', letters);
      }
      const typeIds = aircraftTypeIdsForIcaoLetters(letters);
      if (state.selectedObject && state.selectedObject.type === 'pbb') {
        const pbb = state.selectedObject.obj;
        pbb.categoryMode = 'icao';
        pbb.allowedIcaoCategories = letters;
        pbb.category = representativeCategoryFromLetters(letters);
        pbb.allowedAircraftTypes = typeIds;
        renderAircraftConstraintChoices('standAircraftAccess', typeIds, letters);
        rebuildPbbBridgeGeometry(pbb);
        updateObjectInfo();
        renderObjectList();
        draw();
        update3DSceneWhenVisible();
      } else {
        renderAircraftConstraintChoices('standAircraftAccess', typeIds, letters);
      }
    });
  }
  const pbbLengthInputEl = document.getElementById('pbbLength');
  if (pbbLengthInputEl) {
    pbbLengthInputEl.addEventListener('change', function() {
      const requested = Number(this.value);
      const nextLen = (isFinite(requested) && requested > 0) ? requested : 15;
      this.value = String(Math.max(1, Math.round(nextLen)));
      if (state.selectedObject && state.selectedObject.type === 'pbb') {
        const pbb = state.selectedObject.obj;
        pbb.pbbArmLenM = nextLen;
        applyPbbArmLengthToBridgeEnds(pbb, nextLen);
        updateObjectInfo();
        renderObjectList();
        draw();
        update3DSceneWhenVisible();
      }
    });
  }
  const standAngleInputEl = document.getElementById('standAngle');
  if (standAngleInputEl) {
    standAngleInputEl.addEventListener('change', function() {
      const nextDeg = normalizeAngleDeg(this.value);
      this.value = String(Math.round(nextDeg));
      if (state.selectedObject && state.selectedObject.type === 'pbb') {
        const pbb = state.selectedObject.obj;
        pbb.angleDeg = nextDeg;
        updateObjectInfo();
        renderObjectList();
        draw();
        update3DSceneWhenVisible();
      }
    });
  }
  const pbbBridgeCountInputEl = document.getElementById('pbbBridgeCount');
  if (pbbBridgeCountInputEl) {
    pbbBridgeCountInputEl.addEventListener('change', function() {
      const nextCount = Math.max(1, Math.min(8, parseInt(this.value, 10) || 1));
      this.value = String(nextCount);
      if (state.selectedObject && state.selectedObject.type === 'pbb') {
        const pbb = state.selectedObject.obj;
        pbb.pbbCount = nextCount;
        delete pbb.pbbBridges;
        rebuildPbbBridgeGeometry(pbb);
        updateObjectInfo();
        renderObjectList();
        draw();
        update3DSceneWhenVisible();
      }
    });
  }
  function applyPbbBoardingAreaDimsFromInputs(pbb) {
    const wEl = document.getElementById('pbbBoardingWidth');
    const hEl = document.getElementById('pbbBoardingHeight');
    const nw = Math.max(0.5, Number(wEl && wEl.value) || 5);
    const nh = Math.max(0.5, Number(hEl && hEl.value) || 15);
    pbb.boardingWidthM = nw;
    pbb.boardingHeightM = nh;
    if (wEl) wEl.value = String(nw);
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
          delete tw.queueFlow;
        } else if (ptCur === 'runway_exit' || ptCur === 'runway_taxiway') {
          const kindRx = String(this.value || 'queue');
          if (kindRx === 'normal') tw.queueFlow = false;
          else delete tw.queueFlow;
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

  function syncAllocGanttSimPlayheadPosition() {
    const ganttWrap = document.getElementById('allocationGantt');
    if (!ganttWrap) return;
    const phEl = ganttWrap.querySelector('.alloc-gantt-sim-playhead');
    if (!phEl) return;
    const ctx = state._allocGanttPlayheadCtx;
    if (!ctx || ctx.winStart == null || !isFinite(ctx.winStart)) {
      phEl.style.display = 'none';
      return;
    }
    const simMin = state.simTimeSec / 60;
    if (!isFinite(simMin)) {
      phEl.style.display = 'none';
      return;
    }
    if (simMin < ctx.winStart - 1e-9 || simMin > ctx.winEnd + 1e-9) {
      phEl.style.display = 'none';
      return;
    }
    phEl.style.display = '';
    const leftPct = ((simMin - ctx.winStart) / ctx.displaySpan) * 100 * ctx.zoom;
    phEl.style.left = leftPct + '%';
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
    syncAllocGanttSimPlayheadPosition();
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
    syncAllocGanttSimPlayheadPosition();
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
    const ps = base.split('-');
    const Y = parseInt(ps[0], 10);
    const Mo = parseInt(ps[1], 10) - 1;
    const D = parseInt(ps[2], 10);
    if (!isFinite(Y) || !isFinite(Mo) || !isFinite(D)) return formatMinutesToHHMMSS(minsRaw);
    const t0 = new Date(Y, Mo, D, 0, 0, 0);
    t0.setTime(t0.getTime() + sec * 1000);
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
      let dx = 1, dy = 0;
      if (tr) {
        const fb = playbackLastMotionUnitDirBeforeTime(tr, tSec);
        if (fb) { dx = fb.dx; dy = fb.dy; }
      }
      return { x: a.x, y: a.y, dx: dx, dy: dy, deadlockGhost: dg };
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
    function headingForInterval(i) {
      const a = tl[i], b = tl[i + 1];
      const dx = b.x - a.x, dy = b.y - a.y;
      const l2 = dx * dx + dy * dy;
      if (l2 >= motionChordEps2) return { dx: dx, dy: dy };
      const prev = lastMotionUnitDirBefore(i);
      if (prev) return { dx: prev.dx, dy: prev.dy };
      return { dx: 1, dy: 0 };
    }
    function frBicyclePose(R, x, y, lenM, bmin, dg) {
      if (!R || lenM <= 1e-6) return null;
      const vdx = x - R.x, vdy = y - R.y, vl = Math.hypot(vdx, vdy);
      if (vl < bmin) return null;
      return { x, y, dx: vdx / vl, dy: vdy / vl, deadlockGhost: dg };
    }
    function normHeadingVec(h) {
      const hl = Math.hypot(h.dx, h.dy);
      if (hl < 1e-9) return { dx: 1, dy: 0 };
      return { dx: h.dx / hl, dy: h.dy / hl };
    }
    function segmentIsDghostPair(p, q) {
      return !!(p && q && p.deadlockGhost === true && q.deadlockGhost === true);
    }
    function lastNonDghostMotionUnitDirBeforeEnd(endSegExclusive) {
      for (let j = endSegExclusive - 1; j >= 0; j--) {
        const p = tl[j], q = tl[j + 1];
        if (segmentIsDghostPair(p, q)) continue;
        const u = segmentUnitDir(j);
        if (u) return u;
      }
      return null;
    }
    function firstNonDghostMotionUnitDirFrom(startSeg) {
      for (let j = startSeg; j <= tl.length - 2; j++) {
        const p = tl[j], q = tl[j + 1];
        if (segmentIsDghostPair(p, q)) continue;
        const u = segmentUnitDir(j);
        if (u) return u;
      }
      return null;
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
      const va = Number(a.v), vb = Number(b.v);
      const vThreshMps = 0.05;
      const velocityStill = isFinite(va) && isFinite(vb) && va <= vThreshMps && vb <= vThreshMps;
      let h = headingForInterval(useI);
      if (tr && !lastMotionUnitDirBefore(useI)) {
        const fb = playbackLastMotionUnitDirBeforeTime(tr, tSec);
        if (fb) h = { dx: fb.dx, dy: fb.dy };
      }
      const dg = !!(a.deadlockGhost || b.deadlockGhost);
      let hDraw = h;
      if (dg) {
        const live =
          lastNonDghostMotionUnitDirBeforeEnd(useI + 1) ||
          firstNonDghostMotionUnitDirFrom(useI + 1);
        if (live) hDraw = { dx: live.dx, dy: live.dy };
      }
      if (!dg && velocityStill) {
        let back = lastMotionUnitDirBefore(useI) || lastNonDghostMotionUnitDirBeforeEnd(useI + 1);
        if (!back && tr) back = playbackLastMotionUnitDirBeforeTime(tr, tSec);
        if (back) hDraw = { dx: back.dx, dy: back.dy };
      }
      const hn = normHeadingVec(hDraw);
      const dxAB = b.x - a.x, dyAB = b.y - a.y;
      const dist2 = dxAB * dxAB + dyAB * dyAB;
      const geomStill = dist2 < motionChordEps2;
      const stationary = geomStill || dg || velocityStill;
      const { lenM } = getSimAircraftWorldDimsM(flight);
      const wheelBaseM = 0.55 * lenM;
      const bicycleMin = Math.max(0.15 * motionChordEps, 0.005 * lenM, 0.04);
      let out;
      if (stationary) {
        out = { x, y, dx: hn.dx, dy: hn.dy, deadlockGhost: dg };
      } else {
        out = frBicyclePose(
          walkTimelinePolylineFromPoint(tl, useI, x, y, wheelBaseM, false), x, y, lenM, bicycleMin, dg);
        if (!out) {
          out = { x, y, dx: hn.dx, dy: hn.dy, deadlockGhost: dg };
        }
      }
      return out;
    }
    return null;
  }

  function getPushbackReversePoseForDraw(flight, tSec, pose) {
    if (!pose || !flight) return pose;
    const tr = compactPlaybackTrackForFlight(flight);
    const tl = tr ? compactPlaybackTimelineWindow(tr, tSec, 80) : flight.timeline;
    if (!tl || tl.length < 2) return pose;
    const t = Number(tSec);
    if (!isFinite(t)) return pose;
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
    const segA = tl[segIdx];
    const segB = tl[segIdx + 1];
    if (String(segA.phase || '') !== 'Pushback') return pose;
    let sdx = segB.x - segA.x;
    let sdy = segB.y - segA.y;
    let sl = Math.hypot(sdx, sdy);
    if (sl < 0.08) {
      for (let j = segIdx - 1; j >= 0; j--) {
        const p = tl[j], q = tl[j + 1];
        if (String(p.phase || '') !== 'Pushback') continue;
        const px = q.x - p.x, py = q.y - p.y;
        const pl = Math.hypot(px, py);
        if (pl >= 0.08) { sdx = px; sdy = py; sl = pl; break; }
      }
      for (let j = segIdx + 1; sl < 0.08 && j < tl.length - 1; j++) {
        const p = tl[j], q = tl[j + 1];
        if (String(p.phase || '') !== 'Pushback') continue;
        const px = q.x - p.x, py = q.y - p.y;
        const pl = Math.hypot(px, py);
        if (pl >= 0.08) { sdx = px; sdy = py; sl = pl; break; }
      }
    }
    if (sl < 0.08) return pose;
    return {
      x: pose.x,
      y: pose.y,
      dx: -sdx / sl,
      dy: -sdy / sl,
      deadlockGhost: !!pose.deadlockGhost,
    };
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
    let pose = getFlightPoseAtTime(flight, t);
    if (!pose) return null;
    pose = getPushbackReversePoseForDraw(flight, t, pose);
    pose = applyParkedStandHeadingToPoseIfNeeded(flight, t, pose);
    return pose;
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
    const pFwX = nX * scaleX - 0.15 * dimsM.lenM;
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
  function standIdForParkedApronInterval(f, tSec) {
    const m = f && f.timeline_meta;
    const t = Number(tSec);
    if (!m || !isFinite(t)) return null;
    const eibtList = Array.isArray(m.eibtSecList) ? m.eibtSecList : (typeof m.eibtSec === 'number' ? [m.eibtSec] : []);
    const eobtList = Array.isArray(m.eobtSecList) ? m.eobtSecList : (typeof m.eobtSec === 'number' ? [m.eobtSec] : []);
    const nInt = Math.min(eibtList.length, eobtList.length);
    let idx = -1;
    for (let i = 0; i < nInt; i++) {
      const a = Number(eibtList[i]), b = Number(eobtList[i]);
      if (isFinite(a) && isFinite(b) && t >= a - 1e-3 && t <= b + 1e-3) {
        idx = i;
        break;
      }
    }
    if (idx < 0) return null;
    const segs = Array.isArray(f.apronStaySegments) ? f.apronStaySegments : [];
    if (segs.length > idx && segs[idx] && segs[idx].standId != null && String(segs[idx].standId).trim() !== '') {
      return String(segs[idx].standId);
    }
    if (f.standId != null && String(f.standId).trim() !== '') return String(f.standId);
    return null;
  }
  /**
   * On-block dwell (EIBT–EOBT) with no motion: align nose opposite stand layout axis (+180°) so parked
   * silhouette matches nose-out / drawing convention vs anchor→connection geometry.
   */
  function applyParkedStandHeadingToPoseIfNeeded(flight, tSec, pose) {
    if (!pose || !flight) return pose;
    if (pose.deadlockGhost === true) return pose;
    if (!isFlightParkedAtSimTime(flight, tSec)) return pose;
    if (!isFlightTimelineStationaryAtSimTime(flight, tSec)) return pose;
    const sid = standIdForParkedApronInterval(flight, tSec);
    if (!sid || typeof findStandById !== 'function') return pose;
    const stand = findStandById(sid);
    if (!stand) return pose;
    const id = String(stand.id || '');
    const isPbb = (state.pbbStands || []).some(function(s) { return s && String(s.id) === id; });
    const ang = isPbb ? getPBBStandAngle(stand) : getRemoteStandAngleRad(stand);
    if (!isFinite(ang)) return pose;
    const dx = -Math.cos(ang), dy = -Math.sin(ang);
    return { x: pose.x, y: pose.y, dx: dx, dy: dy, deadlockGhost: !!pose.deadlockGhost };
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
    const laInp = document.getElementById('flightLookaheadTaxiInput');
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
    if (laInp) {
      if (!f) {
        laInp.value = '9';
      } else {
        let v = f.lookaheadTaxi;
        if (v == null || v === '' || !isFinite(Number(v))) v = 9;
        else v = Math.max(0, Math.min(200, Math.floor(Number(v))));
        laInp.value = String(v);
      }
    }
  }
  function syncFlightAssignStrip() {
    const arrEl = document.getElementById('flightAssignStripArr');
    const arrTermEl = document.getElementById('flightAssignStripArrTerm');
    const depTermEl = document.getElementById('flightAssignStripDepTerm');
    const depEl = document.getElementById('flightAssignStripDep');
    const intDomEl = document.getElementById('flightAssignStripIntDom');
    const laInp = document.getElementById('flightLookaheadTaxiInput');
    const sel = state.selectedObject;
    const hasFlight = sel && sel.type === 'flight' && sel.id;
    const f = hasFlight ? state.flights.find(function(x) { return x.id === sel.id; }) : null;
    const dis = !f;
    [arrEl, arrTermEl, depTermEl, depEl, intDomEl, laInp].forEach(function(el) {
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

  /** Flight schedule dynamic AP columns: 10 fixed cells, AP cells, Lookahead_taxi, Dep Rw, then S/E groups. */
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
    const base = apStart + n + 2;
    if (field === 'ap') return apStart;
    if (field === 'lookaheadTaxi') return apStart + n;
    if (field === 'depRunway') return apStart + n + 1;
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
    const laInp0 = document.getElementById('flightLookaheadTaxiInput');
    if (laInp0 && !laInp0._lookaheadTaxiWired) {
      laInp0._lookaheadTaxiWired = true;
      laInp0.addEventListener('change', function() {
        if (!state.selectedObject || state.selectedObject.type !== 'flight') return;
        const f = state.selectedObject.obj;
        let v = parseInt(String(this.value != null ? this.value : '9'), 10);
        if (!isFinite(v)) v = 9;
        v = Math.max(0, Math.min(200, v));
        f.lookaheadTaxi = v;
        this.value = String(v);
        if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
        if (typeof renderFlightList === 'function')
          renderFlightList(false, false, { scheduleMode: 'incremental', dirtyFlightIds: [f.id], touchedStandIds: f.standId ? [f.standId] : [] });
      });
    }
  }

  function _flightListSortedFlightsCopy() {
    const flightsSorted = state.flights.slice();
    flightsSorted.sort(function(a, b) {
      return (a.sibtMin != null ? a.sibtMin : (a.timeMin != null ? a.timeMin : 0)) -
        (b.sibtMin != null ? b.sibtMin : (b.timeMin != null ? b.timeMin : 0));
    });
    return flightsSorted;
  }
  function _flightListSortedIndexForFlightId(flightsSorted, flightId) {
    const want = String(flightId);
    for (let i = 0; i < flightsSorted.length; i++) {
      const f = flightsSorted[i];
      if (f && String(f.id) === want) return i;
    }
    return -1;
  }
  /** Match Flight Schedule row highlight (purple) to ``state.selectedObject`` flight; optional scroll when ``scrollRow``. */
  function _flightListApplyScheduleSelectionHighlightDom(listEl, scrollRow) {
    if (!listEl) return;
    listEl.querySelectorAll('.flight-schedule-table tbody tr.obj-item').forEach(function(r) {
      r.classList.remove('selected', 'expanded');
    });
    const sel = state.selectedObject;
    if (!sel || sel.type !== 'flight' || sel.id == null) return;
    const row = listEl.querySelector('.flight-schedule-table tbody tr.obj-item[data-id="' + String(sel.id) + '"]');
    if (!row) return;
    row.classList.add('selected', 'expanded');
    if (scrollRow) {
      try {
        row.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
      } catch (eScroll) {
        row.scrollIntoView(false);
      }
    }
  }
  /** Grid / external selection: jump pager & virtual scroll so the flight row exists, then highlight. */
  function syncFlightScheduleTableSelectionHighlight() {
    const listEl = document.getElementById('flightList');
    if (!listEl) return;
    const sel = state.selectedObject;
    if (!sel || sel.type !== 'flight' || sel.id == null) {
      _flightListApplyScheduleSelectionHighlightDom(listEl, false);
      return;
    }
    const flightsSorted = _flightListSortedFlightsCopy();
    const idx = _flightListSortedIndexForFlightId(flightsSorted, sel.id);
    if (idx < 0) {
      _flightListApplyScheduleSelectionHighlightDom(listEl, false);
      return;
    }
    const size = FLIGHT_SCHED_PAGE_SIZE;
    const usePagination = size > 0;
    if (usePagination) {
      const targetPage = Math.floor(idx / size);
      if (state.flightSchedulePage !== targetPage) {
        state.flightSchedulePage = targetPage;
        if (typeof renderFlightList === 'function')
          renderFlightList(false, false, { pageTurnOnly: true });
        _flightListApplyScheduleSelectionHighlightDom(listEl, true);
        return;
      }
    }
    const vs = listEl._flightVirtState;
    if (vs && flightsSorted.length && !usePagination) {
      const rowH = vs.rowH || DOM_OPT_FLIGHT_VIRT_ROW_H;
      const vh = listEl.clientHeight || 418;
      listEl.scrollTop = Math.max(0, idx * rowH - Math.max(0, (vh - rowH) * 0.5));
      _flightListPaintVirtualSlice(listEl);
      _flightListApplyScheduleSelectionHighlightDom(listEl, true);
      return;
    }
    _flightListApplyScheduleSelectionHighlightDom(listEl, true);
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
    _flightListApplyScheduleSelectionHighlightDom(listEl, false);
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
