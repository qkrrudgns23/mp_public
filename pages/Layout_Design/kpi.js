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
  document.getElementById('standCategory').addEventListener('change', function() {
    const val = this.value || 'C';
    if (state.selectedObject && state.selectedObject.type === 'pbb') {
      state.selectedObject.obj.category = val;
      rebuildPbbBridgeGeometry(state.selectedObject.obj);
      updateObjectInfo();
      renderObjectList();
      draw();
      update3DSceneWhenVisible();
    }
  });
  const standCategoryModeEl = document.getElementById('standCategoryMode');
  if (standCategoryModeEl) {
    standCategoryModeEl.addEventListener('change', function() {
      syncStandConstraintVisibility('stand', this.value);
      if (state.selectedObject && state.selectedObject.type === 'pbb') {
        state.selectedObject.obj.categoryMode = normalizeStandCategoryMode(this.value, _pbbTier.defaultCategoryMode || 'icao');
        updateObjectInfo();
        renderObjectList();
        draw();
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
      state.selectedObject.obj.allowedAircraftTypes = readCheckedDataItemIds(standAircraftAccessEl, '.aircraft-type-check');
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
  const remoteCategorySelect = document.getElementById('remoteCategory');
  if (remoteCategorySelect) {
    remoteCategorySelect.addEventListener('change', function() {
      const val = this.value || 'C';
      if (state.selectedObject && state.selectedObject.type === 'remote') {
        state.selectedObject.obj.category = val;
        updateObjectInfo();
        renderObjectList();
        draw();
        update3DSceneWhenVisible();
      }
    });
  }
  const remoteCategoryModeEl = document.getElementById('remoteCategoryMode');
  if (remoteCategoryModeEl) {
    remoteCategoryModeEl.addEventListener('change', function() {
      syncStandConstraintVisibility('remote', this.value);
      if (state.selectedObject && state.selectedObject.type === 'remote') {
        state.selectedObject.obj.categoryMode = normalizeStandCategoryMode(this.value, _remoteTier.defaultCategoryMode || 'icao');
        updateObjectInfo();
        renderObjectList();
        draw();
      }
    });
  }
  const remoteAngleInput = document.getElementById('remoteAngle');
  if (remoteAngleInput) {
    remoteAngleInput.addEventListener('change', function() {
      const nextDeg = normalizeAngleDeg(this.value);
      this.value = String(Math.round(nextDeg));
      if (state.selectedObject && state.selectedObject.type === 'remote') {
        state.selectedObject.obj.angleDeg = nextDeg;
        updateObjectInfo();
        renderObjectList();
        draw();
        update3DSceneWhenVisible();
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
      state.selectedObject.obj.allowedAircraftTypes = readCheckedDataItemIds(remoteAircraftAccessEl, '.aircraft-type-check');
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
  const tempStandCategorySelect = document.getElementById('tempStandCategory');
  if (tempStandCategorySelect) {
    tempStandCategorySelect.addEventListener('change', function() {
      const val = this.value || 'C';
      if (state.selectedObject && state.selectedObject.type === 'tempStand') {
        state.selectedObject.obj.category = val;
        updateObjectInfo();
        renderObjectList();
        draw();
        update3DSceneWhenVisible();
      }
    });
  }
  const tempStandCategoryModeEl = document.getElementById('tempStandCategoryMode');
  if (tempStandCategoryModeEl) {
    tempStandCategoryModeEl.addEventListener('change', function() {
      syncStandConstraintVisibility('tempStand', this.value);
      if (state.selectedObject && state.selectedObject.type === 'tempStand') {
        state.selectedObject.obj.categoryMode = normalizeStandCategoryMode(this.value, _remoteTier.defaultCategoryMode || 'icao');
        updateObjectInfo();
        renderObjectList();
        draw();
      }
    });
  }
  const tempStandAngleInput = document.getElementById('tempStandAngle');
  if (tempStandAngleInput) {
    tempStandAngleInput.addEventListener('change', function() {
