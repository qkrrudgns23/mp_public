          tw.queueFlow = kindR === 'queue';
        }
      }
      if (el('taxiwayAvgMoveVelocity')) {
        var v = Number(el('taxiwayAvgMoveVelocity').value);
        tw.avgMoveVelocity = (typeof v === 'number' && isFinite(v) && v > 0) ? Math.max(1, Math.min(50, v)) : 10;
      }
      if (el('runwayMinArrVelocity')) {
        const mav = Number(el('runwayMinArrVelocity').value);
        if (tw.pathType === 'runway') {
          tw.minArrVelocity = (typeof mav === 'number' && isFinite(mav) && mav > 0) ? Math.max(1, Math.min(150, mav)) : 15;
        } else {
          delete tw.minArrVelocity;
        }
      }
      if (tw.pathType === 'runway') {
        const cwEl = el('runwayLineupDistM_CW');
        const ccwEl = el('runwayLineupDistM_CCW');
        const lxCw = cwEl ? Number(cwEl.value) : NaN;
        const lxCcw = ccwEl ? Number(ccwEl.value) : NaN;
        tw.lineupDistM_CW = (typeof lxCw === 'number' && isFinite(lxCw) && lxCw >= 0) ? lxCw : 0;
        tw.lineupDistM_CCW = (typeof lxCcw === 'number' && isFinite(lxCcw) && lxCcw >= 0) ? lxCcw : 0;
        tw.lineupDistM = getEffectiveRunwayLineupDistM(tw);
      } else if (tw.pathType !== 'runway') {
        delete tw.lineupDistM;
        delete tw.lineupDistM_CW;
        delete tw.lineupDistM_CCW;
      }
      if (tw.pathType === 'runway') {
        const startDisp = Number(el('runwayStartDisplacedThresholdM') ? el('runwayStartDisplacedThresholdM').value : RUNWAY_START_DISPLACED_THRESHOLD_DEFAULT_M);
        const startBlast = Number(el('runwayStartBlastPadM') ? el('runwayStartBlastPadM').value : RUNWAY_START_BLAST_PAD_DEFAULT_M);
        const endDisp = Number(el('runwayEndDisplacedThresholdM') ? el('runwayEndDisplacedThresholdM').value : RUNWAY_END_DISPLACED_THRESHOLD_DEFAULT_M);
        const endBlast = Number(el('runwayEndBlastPadM') ? el('runwayEndBlastPadM').value : RUNWAY_END_BLAST_PAD_DEFAULT_M);
        tw.startDisplacedThresholdM = (typeof startDisp === 'number' && isFinite(startDisp) && startDisp >= 0) ? startDisp : RUNWAY_START_DISPLACED_THRESHOLD_DEFAULT_M;
        tw.startBlastPadM = (typeof startBlast === 'number' && isFinite(startBlast) && startBlast >= 0) ? startBlast : RUNWAY_START_BLAST_PAD_DEFAULT_M;
        tw.endDisplacedThresholdM = (typeof endDisp === 'number' && isFinite(endDisp) && endDisp >= 0) ? endDisp : RUNWAY_END_DISPLACED_THRESHOLD_DEFAULT_M;
        tw.endBlastPadM = (typeof endBlast === 'number' && isFinite(endBlast) && endBlast >= 0) ? endBlast : RUNWAY_END_BLAST_PAD_DEFAULT_M;
      } else {
        delete tw.startDisplacedThresholdM;
        delete tw.startBlastPadM;
        delete tw.endDisplacedThresholdM;
        delete tw.endBlastPadM;
      }
      if (tw.pathType !== 'runway_exit' && tw.pathType !== 'runway_taxiway') delete tw.queueFlow;
    }
  }

  function syncSettingsPaneToMode() {
    const mode = settingModeSelect ? settingModeSelect.value : 'grid';
    if (layoutModeTabs) {
      layoutModeTabs.querySelectorAll('.layout-mode-tab').forEach(function(btn) {
        btn.classList.toggle('active', btn.getAttribute('data-mode') === mode);
      });
    }
    document.querySelectorAll('.settings-pane').forEach(el => { el.style.display = 'none'; });
    const paneKey = isPathLayoutMode(mode) ? 'taxiway' : mode;
    const pane = document.getElementById('settings-' + paneKey);
    if (pane) pane.style.display = 'block';
    if (mode === 'marker') {
      syncMarkerFlightAircraftRowVisibility();
      syncMarkerIslandWidthRowVisibility();
      syncMarkerNavaidRowVisibility();
    }
    if (isPathLayoutMode(mode)) {
      const pt = pathTypeFromLayoutMode(mode);
      syncPathFieldVisibilityForPathType(pt);
      if (!state.selectedObject || state.selectedObject.type !== 'taxiway') {
        const nameInput = document.getElementById('taxiwayName');
        if (nameInput) nameInput.value = '';
        const widthInput = document.getElementById('taxiwayWidth');
        if (widthInput) {
          widthInput.value = pt === 'runway'
            ? RUNWAY_PATH_DEFAULT_WIDTH
            : (pt === 'runway_exit' ? RUNWAY_EXIT_DEFAULT_WIDTH : TAXIWAY_DEFAULT_WIDTH);
        }
        syncPathPavementRadiosToValue(pathPavementDefaultForPathType(pt));
        if (pt === 'runway') {
          const startDispInput = document.getElementById('runwayStartDisplacedThresholdM');
          if (startDispInput) startDispInput.value = String(RUNWAY_START_DISPLACED_THRESHOLD_DEFAULT_M);
          const startBlastInput = document.getElementById('runwayStartBlastPadM');
          if (startBlastInput) startBlastInput.value = String(RUNWAY_START_BLAST_PAD_DEFAULT_M);
          const endDispInput = document.getElementById('runwayEndDisplacedThresholdM');
          if (endDispInput) endDispInput.value = String(RUNWAY_END_DISPLACED_THRESHOLD_DEFAULT_M);
          const endBlastInput = document.getElementById('runwayEndBlastPadM');
          if (endBlastInput) endBlastInput.value = String(RUNWAY_END_BLAST_PAD_DEFAULT_M);
        }
        const pathKindIdleSt = document.getElementById('taxiwayPathTypeKind');
        if (pathKindIdleSt && (pt === 'runway_exit' || pt === 'runway_taxiway')) pathKindIdleSt.value = 'normal';
      }
    }
    if (typeof renderObjectList === 'function') renderObjectList();
  }

  settingModeSelect.addEventListener('change', function() {
    cancelActiveLayoutDrawingState();
    state.selectedObject = null;
    syncSettingsPaneToMode();
  });
  if (layoutModeTabs && settingModeSelect) {
    layoutModeTabs.querySelectorAll('.layout-mode-tab').forEach(function(btn) {
      btn.addEventListener('click', function() {
        const mode = this.getAttribute('data-mode') || 'grid';
        if (settingModeSelect.value === mode) {
          cancelActiveLayoutDrawingState();
          syncSettingsPaneToMode();
          return;
        }
        settingModeSelect.value = mode;
        settingModeSelect.dispatchEvent(new Event('change'));
      });
    });
  }
  syncSettingsPaneToMode();

  let activeTab = 'settings';
  function switchToTab(tabId) {
    activeTab = tabId;
    cancelActiveLayoutDrawingState();
    document.querySelectorAll('.right-panel-tab').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    const tabBtn = document.querySelector('.right-panel-tab[data-tab="' + tabId + '"]');
    const tabEl = document.getElementById('tab-' + tabId);
    if (tabBtn) tabBtn.classList.add('active');
    if (tabEl) tabEl.classList.add('active');
    if (tabId === 'flight') {
      if (state.selectedObject && state.selectedObject.type === 'flight' && typeof hookSyncFlightPanelFromSelection === 'function')
        hookSyncFlightPanelFromSelection();
      if (typeof renderFlightList === 'function') {
        const flightListEl = document.getElementById('flightList');
        const needsRerender = !flightListEl || !flightListEl.querySelector('.flight-schedule-table tbody tr:not(.flight-virt-spacer)');
        if (needsRerender) renderFlightList();
      }
    }
    if (tabId === 'allocation' && typeof renderFlightGantt === 'function') renderFlightGantt({ skipPathPrep: true });
    if (tabId === 'rwysep') {
      const rwyPanel = document.getElementById('rwySepPanel');
      if (
        state.rwySepPanelDirty === false &&
        rwyPanel &&
        document.getElementById('rwysep-standard') &&
        typeof drawRwySeparationTimeline === 'function'
      ) {
        drawRwySeparationTimeline(rwyPanel);
      } else if (typeof renderRunwaySeparation === 'function') {
        renderRunwaySeparation();
      }
    }
  }
  document.querySelectorAll('.right-panel-tab').forEach(btn => {
    btn.addEventListener('click', function() { switchToTab(this.getAttribute('data-tab')); });
  });

  ['chkShowSPoints', 'chkShowEBar', 'chkShowEPoints', 'chkShowSBars'].forEach(function(chkId) {
    const el = document.getElementById(chkId);
    if (el) el.addEventListener('change', function() {
      if (typeof renderFlightGantt === 'function') renderFlightGantt({ skipPathPrep: true });
    });
  });

  document.getElementById('gridCellSize').addEventListener('change', function() { CELL_SIZE = Math.max(5, Number(this.value) || 5); invalidateGridUnderlay(); draw(); });
  document.getElementById('gridCols').addEventListener('change', function() { GRID_COLS = Math.max(5, Math.min(1000, parseInt(this.value,10)||400)); invalidateGridUnderlay(); draw(); });
  document.getElementById('gridRows').addEventListener('change', function() { GRID_ROWS = Math.max(5, Math.min(1000, parseInt(this.value,10)||400)); invalidateGridUnderlay(); draw(); });
  function commitGridLayoutImageNumericChange(inputId, applyFn) {
    const input = document.getElementById(inputId);
    if (!input) return;
    input.addEventListener('change', function() {
      if (!state.layoutImageOverlay) {
        syncPanelFromState();
        return;
      }
      const before = JSON.stringify(state.layoutImageOverlay);
      const snapshot = JSON.parse(before);
      applyFn(this);
      const after = JSON.stringify(state.layoutImageOverlay);
      if (before === after) {
        syncPanelFromState();
        invalidateGridUnderlay();
        draw();
        return;
      }
      undoStack.push({
        terminals: JSON.parse(JSON.stringify(state.terminals || [])),
        pbbStands: JSON.parse(JSON.stringify(state.pbbStands || [])),
        remoteStands: JSON.parse(JSON.stringify(state.remoteStands || [])),
        tempStands: JSON.parse(JSON.stringify(state.tempStands || [])),
        holdingPoints: JSON.parse(JSON.stringify(state.holdingPoints || [])),
        taxiways: JSON.parse(JSON.stringify(state.taxiways || [])),
        apronLinks: JSON.parse(JSON.stringify(state.apronLinks || [])),
        layoutImageOverlay: snapshot,
        layoutEdgeNames: JSON.parse(JSON.stringify(state.layoutEdgeNames || {})),
        directionModes: JSON.parse(JSON.stringify(state.directionModes || [])),
        flights: cloneFlightsWithoutPathPolylineCache(state.flights),
        layoutMarkers: JSON.parse(JSON.stringify(state.layoutMarkers || []))
      });
      if (undoStack.length > maxUndoLevels) undoStack.shift();
      syncPanelFromState();
      invalidateGridUnderlay();
      draw();
    });
  }
  const gridLayoutImageFileEl = document.getElementById('gridLayoutImageFile');
  if (gridLayoutImageFileEl) {
    gridLayoutImageFileEl.addEventListener('change', function() {
      const file = this.files && this.files[0];
      if (!file) return;
      const fileType = String(file.type || '').toLowerCase();
      const fileName = String(file.name || 'Layout image');
      const accepted = fileType === 'image/png' || fileType === 'image/jpeg' || fileType === 'image/svg+xml' ||
        /\.(png|jpe?g|svg)$/i.test(fileName);
      if (!accepted) {
        alert('Only PNG, JPG, JPEG, and SVG files are supported.');
        this.value = '';
        return;
      }
      const reader = new FileReader();
      reader.onload = function(ev) {
        const dataUrl = ev && ev.target ? String(ev.target.result || '') : '';
        if (!dataUrl) return;
        const img = new Image();
        img.onload = function() {
          const widthM = state.layoutImageOverlay ? clampLayoutImageSize(state.layoutImageOverlay.widthM, GRID_LAYOUT_IMAGE_DEFAULTS.widthM) : GRID_LAYOUT_IMAGE_DEFAULTS.widthM;
          const aspect = (img.naturalWidth > 0 && img.naturalHeight > 0)
            ? (img.naturalHeight / img.naturalWidth)
            : (GRID_LAYOUT_IMAGE_DEFAULTS.heightM / Math.max(GRID_LAYOUT_IMAGE_DEFAULTS.widthM, 1e-9));
          const heightM = state.layoutImageOverlay
            ? clampLayoutImageSize(state.layoutImageOverlay.heightM, Math.max(1, widthM * aspect))
            : Math.max(1, widthM * aspect);
          pushUndo();
          state.layoutImageOverlay = normalizeLayoutImageOverlay({
            name: fileName,
            type: fileType || 'image/png',
            dataUrl: dataUrl,
            opacity: state.layoutImageOverlay ? state.layoutImageOverlay.opacity : GRID_LAYOUT_IMAGE_DEFAULTS.opacity,
            widthM: widthM,
            heightM: heightM,
            originalWidthPx: img.naturalWidth || widthM,
            originalHeightPx: img.naturalHeight || heightM,
            topLeftCol: state.layoutImageOverlay ? state.layoutImageOverlay.topLeftCol : GRID_LAYOUT_IMAGE_DEFAULTS.topLeftCol,
            topLeftRow: state.layoutImageOverlay ? state.layoutImageOverlay.topLeftRow : GRID_LAYOUT_IMAGE_DEFAULTS.topLeftRow
          });
          syncLayoutImageBitmap();
          syncPanelFromState();
          draw();
        };
        img.onerror = function() {
          alert('Failed to read the selected layout image.');
          gridLayoutImageFileEl.value = '';
        };
        img.src = dataUrl;
      };
      reader.readAsDataURL(file);
    });
  }
  const clearGridLayoutImageBtn = document.getElementById('btnClearGridLayoutImage');
  if (clearGridLayoutImageBtn) {
    clearGridLayoutImageBtn.addEventListener('click', function() {
      if (!state.layoutImageOverlay) return;
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
          const kindRx = String(this.value || 'normal');
          tw.queueFlow = kindRx === 'queue';
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

  /** Min SIBT (minutes) / max SOBT (minutes) across apron stay segments / flight fields. */
  function computeFleetSibtSobtMinMaxMinutesAmongFlights() {
    let minSibtM = Infinity;
    let maxSobtM = -Infinity;
    (state.flights || []).forEach(function(f) {
      if (!f) return;
      const segs = typeof normalizeFlightApronStaySegments === 'function' ? normalizeFlightApronStaySegments(f) : [];
      if (segs && segs.length) {
        segs.forEach(function(seg) {
          const s = Number(seg && seg.sibtMin);
          const t = Number(seg && seg.sobtMin);
          if (isFinite(s)) minSibtM = Math.min(minSibtM, s);
          if (isFinite(t)) maxSobtM = Math.max(maxSobtM, t);
        });
      } else {
        const s = f.sibtMin != null ? Number(f.sibtMin) : (f.timeMin != null ? Number(f.timeMin) : NaN);
        let t = f.sobtMin != null ? Number(f.sobtMin) : NaN;
        if (!isFinite(t) && isFinite(s)) t = s + Math.max(0, Number(f.dwellMin) || 45);
        if (isFinite(s)) minSibtM = Math.min(minSibtM, s);
        if (isFinite(t)) maxSobtM = Math.max(maxSobtM, t);
      }
    });
    return { minSibtM: minSibtM, maxSobtM: maxSobtM };
  }

  function getSimAxisLoHiSec() {
    const lo = Number(state.simStartSec), hi = Number(state.simDurationSec);
    return {
      axisLo: isFinite(lo) ? lo : 0,
      axisHi: isFinite(hi) ? hi : 0,
      span: (isFinite(lo) && isFinite(hi) && hi > lo) ? hi - lo : 0,
    };
  }

  function getSimPlaybackWindowLoHiSec() {
    const ax = getSimAxisLoHiSec();
    const axisLo = ax.axisLo, axisHi = ax.axisHi;
    let wLo = Number(state.simWindowStartSec), wHi = Number(state.simWindowEndSec);
    if (!isFinite(wLo) || !isFinite(wHi)) {
      return { lo: axisLo, hi: axisHi, axisLo: axisLo, axisHi: axisHi };
    }
    wLo = Math.max(axisLo, Math.min(axisHi, wLo));
    wHi = Math.max(axisLo, Math.min(axisHi, wHi));
    if (wHi < wLo + SIM_TIME_SLIDER_SNAP_SEC * 0.5) {
      wHi = Math.min(axisHi, wLo + SIM_TIME_SLIDER_SNAP_SEC);
    }
    return { lo: wLo, hi: wHi, axisLo: axisLo, axisHi: axisHi };
  }

  function snapSimTimeToPlaybackWindowSec(tSec) {
    const b = getSimPlaybackWindowLoHiSec();
    const step = SIM_TIME_SLIDER_SNAP_SEC;
    const t = Number(tSec);
    if (!isFinite(t)) return b.lo;
    let clamped = Math.max(b.lo, Math.min(b.hi, t));
    if (!(step > 0)) return clamped;
    let snapped = b.lo + Math.round((clamped - b.lo) / step) * step;
    if (snapped < b.lo) snapped = b.lo;
    if (snapped > b.hi) snapped = b.hi;
    return snapped;
  }

  function flightHasSibtInSimWindowSec(f, wLo, wHi) {
    if (!f || !isFinite(Number(wLo)) || !isFinite(Number(wHi))) return false;
    const segs = typeof normalizeFlightApronStaySegments === 'function' ? normalizeFlightApronStaySegments(f) : [];
    if (segs && segs.length) {
      for (let si = 0; si < segs.length; si++) {
        const s = Number(segs[si].sibtMin);
        if (!isFinite(s)) continue;
        const sSec = s * 60;
        if (sSec >= wLo - 1e-6 && sSec <= wHi + 1e-6) return true;
      }
      return false;
    }
    const s = f.sibtMin != null ? Number(f.sibtMin) : (f.timeMin != null ? Number(f.timeMin) : NaN);
    if (!isFinite(s)) return false;
    const sSec = s * 60;
    return sSec >= wLo - 1e-6 && sSec <= wHi + 1e-6;
  }

  function syncSimPlaybackRangeInputsDom() {
    const ax = getSimAxisLoHiSec();
    const wh = getSimPlaybackWindowLoHiSec();
    const lo = ax.axisLo, hi = ax.axisHi;
    const startEl = document.getElementById('flightSimSliderWindowStart');
    const curEl = document.getElementById('flightSimSlider');
    const endEl = document.getElementById('flightSimSliderWindowEnd');
    const stepStr = String(SIM_TIME_SLIDER_SNAP_SEC);
    [startEl, curEl, endEl].forEach(function(el) {
      if (!el) return;
      el.min = String(lo);
      el.max = String(hi);
      el.step = stepStr;
    });
    if (startEl) startEl.value = String(wh.lo);
    if (endEl) endEl.value = String(wh.hi);
    if (curEl) curEl.value = String(state.simTimeSec);
  }

  try {
    window.__getSimPlaybackWindowLoHiSec = getSimPlaybackWindowLoHiSec;
    window.__snapSimTimeToPlaybackWindowSec = snapSimTimeToPlaybackWindowSec;
    window.__flightHasSibtInSimWindowSec = flightHasSibtInSimWindowSec;
  } catch (__expWin) {
    /* ignore */
  }

  /**
   * 더블 클릭으로 미세 모드 토글 후, 같은 슬라이더에서 포인터 드래그 시 (전체 타임축 폭 대비 픽셀 이동량)×(span)/SIM_TIME_SLIDER_FINE_DIVISOR 만큼만 시각 변경.
   * 중복 초기화 방지 위해 ``dataset.airsideFineBind`` 사용.
   */
  function bindFlightSimSliderFineOnce(sliderEl) {
    if (!sliderEl || sliderEl.dataset.airsideFineBind === '1') return;
    sliderEl.dataset.airsideFineBind = '1';
    let fineDragPid = null;

    sliderEl.addEventListener('dblclick', function(ev) {
      ev.preventDefault();
      state.simTimeSliderFineMode = !state.simTimeSliderFineMode;
      if (state.simTimeSliderFineMode) {
        sliderEl.classList.add('sim-time-slider-fine');
      } else {
        sliderEl.classList.remove('sim-time-slider-fine');
      }
    });

    /** 미세 모드에서 ``pointerdown`` 시 바로 ``preventDefault`` 하면 브라우저가 ``click``/``dblclick`` 을 만들지 않아 꺼짐 토글이 안 될 수 있음 → 임계 이동 후에만 캡처·차단 */
    sliderEl.addEventListener('pointerdown', function(ev) {
      if (!state.simTimeSliderFineMode) return;
      if (ev.button != null && ev.button !== 0) return;
      if (ev.isPrimary === false) return;
      const lo = Number(state.simStartSec);
      const hi = Number(state.simDurationSec);
      if (!isFinite(lo) || !isFinite(hi) || !(hi > lo + 1e-9)) return;
      const wh = typeof getSimPlaybackWindowLoHiSec === 'function' ? getSimPlaybackWindowLoHiSec() : { lo: lo, hi: hi };
      const wLo = Number(wh.lo), wHi = Number(wh.hi);
      const rect = sliderEl.getBoundingClientRect();
      const wTrack = rect.width > 1 ? rect.width : 1;
      const span = Math.max(1e-9, wHi - wLo);
      const startT = typeof snapSimTimeToPlaybackWindowSec === 'function'
        ? snapSimTimeToPlaybackWindowSec(Number(state.simTimeSec))
        : Number(state.simTimeSec);
      const startClientX = ev.clientX;

      fineDragPid = ev.pointerId;
      let dragged = false;

      function onMove(me) {
        if (fineDragPid === null || me.pointerId !== fineDragPid) return;
        const dxAbs = Math.abs(me.clientX - startClientX);
        if (!dragged && dxAbs < 4) return;
        if (!dragged) {
          dragged = true;
          try {
            sliderEl.setPointerCapture(me.pointerId);
          } catch (_) { /* ignore */ }
          me.preventDefault();
          me.stopPropagation();
          state.simSliderScrubbing = true;
        }
        const dx = me.clientX - startClientX;
        const deltaSec = (dx / wTrack) * span / SIM_TIME_SLIDER_FINE_DIVISOR;
        let next = snapSimTimeToPlaybackWindowSec(startT + deltaSec);
        next = Math.max(wLo, Math.min(wHi, next));
        state.simTimeSec = next;
        sliderEl.value = String(next);
        if (typeof updateFlightSimPlaybackLabelsDom === 'function') updateFlightSimPlaybackLabelsDom();
        syncAllocGanttSimPlayheadPosition();
        try { draw({ bypassSimScrubGuard: true }); } catch (_) { /* ignore */ }
        update3DSceneWhenVisible();
      }

      function finish(me) {
        if (fineDragPid === null) return;
        if (me && me.pointerId != null && me.pointerId !== fineDragPid) return;
        document.removeEventListener('pointermove', onMove);
        document.removeEventListener('pointerup', finish, true);
        document.removeEventListener('pointercancel', finish, true);
        if (dragged) {
          try {
            sliderEl.releasePointerCapture(fineDragPid);
          } catch (_) { /* ignore */ }
        }
        fineDragPid = null;
        state.simSliderScrubbing = false;
        if (typeof updateFlightSimPlaybackLabelsDom === 'function') updateFlightSimPlaybackLabelsDom();
        syncAllocGanttSimPlayheadPosition();
        try { draw({ bypassSimScrubGuard: true }); } catch (_) { /* ignore */ }
        update3DSceneWhenVisible();
      }

      document.addEventListener('pointermove', onMove);
      document.addEventListener('pointerup', finish, true);
      document.addEventListener('pointercancel', finish, true);
    }, true);
  }

  try {
    window.__bindFlightSimSliderFineOnce = bindFlightSimSliderFineOnce;
  } catch (_eFineWin) {
    /* ignore */
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
    const fleet = computeFleetSibtSobtMinMaxMinutesAmongFlights();
    let axisMinSec = 0;
    let axisMaxSec = 0;
    if (isFinite(fleet.minSibtM) && isFinite(fleet.maxSobtM) && fleet.maxSobtM >= fleet.minSibtM - 1e-9) {
      axisMinSec = Math.max(0, fleet.minSibtM * 60 - SIM_AXIS_SIBT_BEFORE_SEC);
      axisMaxSec = fleet.maxSobtM * 60 + SIM_AXIS_SOBT_AFTER_SEC;
    }
    if (!(axisMaxSec > axisMinSec + 1e-9)) {
      axisMaxSec = axisMinSec + Math.max(SIM_TIME_SLIDER_SNAP_SEC, 60);
    }
    state.simStartSec = axisMinSec;
    state.simDurationSec = axisMaxSec;
    const nFl = (state.flights || []).length;
    const axisKey =
      String(fleet.minSibtM) + '|' + String(fleet.maxSobtM) + '|' + String(nFl);
    const pv = state._pendingPersistSimWindow;
    const usePersist = !!(pv && isFinite(Number(pv.lo)) && isFinite(Number(pv.hi)));
    if (usePersist) {
      let wLo = Math.max(axisMinSec, Math.min(axisMaxSec, Number(pv.lo)));
      let wHi = Math.max(axisMinSec, Math.min(axisMaxSec, Number(pv.hi)));
      if (wHi < wLo + SIM_TIME_SLIDER_SNAP_SEC * 0.5) {
        wHi = Math.min(axisMaxSec, wLo + SIM_TIME_SLIDER_SNAP_SEC);
      }
      state.simWindowStartSec = snapSimTimeSecForSlider(wLo);
      state.simWindowEndSec = snapSimTimeSecForSlider(wHi);
      state._pendingPersistSimWindow = null;
      state._simScheduleAxisKey = axisKey;
    } else {
      let resetWindow = state._simScheduleAxisKey !== axisKey;
      state._simScheduleAxisKey = axisKey;
      if (resetWindow) {
        state.simWindowStartSec = snapSimTimeSecForSlider(axisMinSec);
        state.simWindowEndSec = snapSimTimeSecForSlider(axisMaxSec);
      } else {
        state.simWindowStartSec = snapSimTimeSecForSlider(
          Math.max(axisMinSec, Math.min(axisMaxSec, Number(state.simWindowStartSec)))
        );
        state.simWindowEndSec = snapSimTimeSecForSlider(
          Math.max(axisMinSec, Math.min(axisMaxSec, Number(state.simWindowEndSec)))
        );
        if (state.simWindowEndSec < state.simWindowStartSec + SIM_TIME_SLIDER_SNAP_SEC * 0.5) {
          state.simWindowEndSec = snapSimTimeSecForSlider(
            Math.min(axisMaxSec, state.simWindowStartSec + SIM_TIME_SLIDER_SNAP_SEC)
          );
        }
      }
    }
    const wh = getSimPlaybackWindowLoHiSec();
    state.simWindowStartSec = snapSimTimeSecForSlider(wh.lo);
    state.simWindowEndSec = snapSimTimeSecForSlider(wh.hi);
    state.simTimeSec = snapSimTimeToPlaybackWindowSec(state.simTimeSec);
    syncSimPlaybackRangeInputsDom();
    const anySlider = document.getElementById('flightSimSlider');
    const axisBad = !(state.simDurationSec > state.simStartSec + 1e-9);
    const sStart = document.getElementById('flightSimSliderWindowStart');
    const sEnd = document.getElementById('flightSimSliderWindowEnd');
    if (anySlider) anySlider.disabled = axisBad;
    if (sStart) sStart.disabled = axisBad;
    if (sEnd) sEnd.disabled = axisBad;
    if (typeof bindFlightSimSliderFineOnce === 'function') bindFlightSimSliderFineOnce(anySlider);
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
    state.simTimeSec = snapSimTimeToPlaybackWindowSec(state.simWindowStartSec);
    if (typeof syncSimPlaybackRangeInputsDom === 'function') syncSimPlaybackRangeInputsDom();
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
      if (!f) continue;
      const m = f.timeline_meta;
      if (m && !m.error) {
        const eibtList = Array.isArray(m.eibtSecList) ? m.eibtSecList : (typeof m.eibtSec === 'number' ? [m.eibtSec] : []);
        const eobtList = Array.isArray(m.eobtSecList) ? m.eobtSecList : (typeof m.eobtSec === 'number' ? [m.eobtSec] : []);
        if (Math.min(eibtList.length, eobtList.length) > 0) {
          const sidOcc = standIdForParkedApronInterval(f, t);
          if (sidOcc) set.add(String(sidOcc));
          continue;
        }
      }
      if (!f.standId) continue;
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
  function flightStandTerminalConstraintsOk(f, stand, segmentIdx, segmentCount) {
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
  function flightCanUseStandForSegment(f, stand, segmentIdx, segmentCount) {
    if (!flightStandAircraftConstraintOk(f, stand)) return false;
    return flightStandTerminalConstraintsOk(f, stand, segmentIdx, segmentCount);
  }
  function flightCanUseStand(f, stand) {
    const segs = typeof normalizeFlightApronStaySegments === 'function' ? normalizeFlightApronStaySegments(f) : [];
    return flightCanUseStandForSegment(f, stand, 0, Math.max(1, segs.length || 1));
  }
  function showAllocationConstraintModal(message, optTitle) {
    const msg = String(message || 'This stand assignment is not allowed.');
    const ttl =
      arguments.length >= 2 &&
      optTitle !== undefined &&
      optTitle !== null &&
      String(optTitle).trim() !== ''
        ? String(optTitle).trim()
        : 'Assignment not allowed';
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
    const titleEl = el.querySelector('.alloc-constraint-modal__title');
    if (titleEl) titleEl.textContent = ttl;
    const msgEl = el.querySelector('.alloc-constraint-modal__message');
    if (msgEl) msgEl.textContent = msg;
    el.classList.add('is-open');
  }

  function assignStandToFlight(f, standId, segmentIdx, opts) {
    if (!f) return false;
    const fromGantt = !!(opts && opts.fromAllocGantt);
    const ganttConstraintMsg =
      'Stand constraints or selected Arr/Dep Building do not match this aircraft, so it cannot be assigned.';
    if (standId) {
      const allStands = allStandsForFlightAssignment();
      const stand = allStands.find(function(s) { return s.id === standId; });
      const segsForValidation = typeof normalizeFlightApronStaySegments === 'function' ? normalizeFlightApronStaySegments(f) : [];
      const segCount = Math.max(1, segsForValidation.length || 1);
      const segIdxForValidation = segmentIdx != null && isFinite(Number(segmentIdx)) ? Math.max(0, parseInt(segmentIdx, 10) || 0) : 0;
      if (!stand) {
        showAllocationConstraintModal(ganttConstraintMsg, 'Assignment not allowed');
        return false;
      }
      if (!flightStandAircraftConstraintOk(f, stand)) {
        if (fromGantt) {
          showAllocationConstraintModal(ganttConstraintMsg, 'Assignment not allowed');
        } else {
          const apronNo = String((stand.name && String(stand.name).trim()) || stand.id || standId || '—').trim();
          const regNo =
            String(
              f.reg != null && String(f.reg).trim() !== ''
                ? f.reg
                : f.flightNumber || f.id || ''
            ).trim() || '—';
          showAllocationConstraintModal(apronNo + '\n' + regNo, 'Apron size 오류발생');
        }
        return false;
      }
      if (!flightStandTerminalConstraintsOk(f, stand, segIdxForValidation, segCount)) {
        showAllocationConstraintModal(ganttConstraintMsg, 'Assignment not allowed');
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
    function lastMotionUnitDirBefore(i, opts) {
      const skipPb = opts && opts.skipPushback === true;
      for (let j = i - 1; j >= 0; j--) {
        if (skipPb && String(tl[j].phase || '') === 'Pushback') continue;
        const u = segmentUnitDir(j);
        if (u) return u;
      }
      return null;
    }
    function firstMotionUnitDirFrom(i, opts) {
      const skipPb = opts && opts.skipPushback === true;
      for (let j = i; j <= tl.length - 2; j++) {
        if (skipPb && String(tl[j].phase || '') === 'Pushback') continue;
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
      const curPb = String(tl[i].phase || '') === 'Pushback';
      const prev = lastMotionUnitDirBefore(i, { skipPushback: !curPb });
      if (prev) return { dx: prev.dx, dy: prev.dy };
      if (!curPb) {
        const fwd = firstMotionUnitDirFrom(i, { skipPushback: true });
        if (fwd) return { dx: fwd.dx, dy: fwd.dy };
      }
      return { dx: 1, dy: 0 };
    }
    function normHeadingVec(h) {
      const hl = Math.hypot(h.dx, h.dy);
      if (hl < 1e-9) return { dx: 1, dy: 0 };
      return { dx: h.dx / hl, dy: h.dy / hl };
    }
    const idxAtTime = timelineSegmentIndexAtTime(tl, tSec, false);
    if (idxAtTime >= 0) {
      const i = idxAtTime;
      const a = tl[i], b = tl[i+1];
      const span = b.t - a.t || 1;
      const u = (tSec - a.t) / span;
      const x = a.x + (b.x - a.x) * u;
      const y = a.y + (b.y - a.y) * u;
      const h = headingForInterval(i);
      const dg = !!(a.deadlockGhost || b.deadlockGhost);
      const hn = normHeadingVec(h);
      return { x: x, y: y, dx: hn.dx, dy: hn.dy, deadlockGhost: dg };
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
    if (String(segB.phase || '') !== 'Pushback') return pose;
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
    if (isFlightAirsideCycleCompleteAtSimTime(flight, t)) {
      if (flight) {
        flight.__parkedStationaryPoseCache = undefined;
        flight.__parkedSilBmpCache = undefined;
      }
      return null;
    }
    const trWin = compactPlaybackTrackStartEnd(compactPlaybackTrackForFlight(flight));
    const tl = flight && flight.timeline;
    const t0 = trWin ? trWin.t0 : (tl && tl.length ? tl[0].t : NaN);
    const t1 = trWin ? trWin.t1 : (tl && tl.length ? tl[tl.length - 1].t : NaN);
    if (!isFinite(t0) || !isFinite(t1)) return null;
    if (t + 1e-9 < t0) return null;
    if (t > t1) t = t1;
    const parkedCtx = getParkedOnBlockStationaryPoseCacheCtx(flight, t);
    if (parkedCtx) {
      const cached = flight.__parkedStationaryPoseCache;
      if (cached && cached.key === parkedCtx.key && cached.pose) {
        return cached.pose;
      }
      let poseP = getFlightPoseAtTime(flight, parkedCtx.anchorT);
      if (!poseP) return null;
      poseP = getPushbackReversePoseForDraw(flight, parkedCtx.anchorT, poseP);
      poseP = applyParkedStandHeadingToPoseIfNeeded(flight, parkedCtx.anchorT, poseP);
      flight.__parkedStationaryPoseCache = { key: parkedCtx.key, pose: poseP };
      return poseP;
    }
    let pose = getFlightPoseAtTime(flight, t);
    if (!pose) return null;
    pose = getPushbackReversePoseForDraw(flight, t, pose);
    pose = applyParkedStandHeadingToPoseIfNeeded(flight, t, pose);
    return pose;
  }
  function simFlightSilhouetteWorldPolygon(f, pose, tSecOpt) {
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
    let fuselageStationFrac = 0.15;
    if (typeof tSecOpt === 'number' && isFinite(tSecOpt) && simFlightPhaseAtTime(f, tSecOpt, pose) === 'Pushback') {
      fuselageStationFrac = 0.25;
    }
    const pFwX = nX * scaleX - fuselageStationFrac * dimsM.lenM;
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
    if (!seg || !seg.a) return '';
    const pa = seg.a.phase != null ? String(seg.a.phase || '') : '';
    const pb = seg.b && seg.b.phase != null ? String(seg.b.phase || '') : pa;
    if (pa === 'Pushback' && pb === 'Pushback') return 'Pushback';
    if (pa === 'Pushback' && pb && pb !== 'Pushback') return pb;
    return pa || pb || '';
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
   * Formerly rotated nose to stand layout axis while on-block stationary; kept as a no-op so heading
   * comes only from timeline/pose logic (no velocity- or dwell-based direction overrides).
   */
  function applyParkedStandHeadingToPoseIfNeeded(flight, tSec, pose) {
    return pose;
  }
  /** EIBT–EOBT on-block stationary: pose unchanged for the dwell; skip repeat getFlightPoseAtTime sampling. */
  function getParkedOnBlockStationaryPoseCacheCtx(flight, tSec) {
    const t = Number(tSec);
    if (!flight || !isFinite(t)) return null;
    if (!isFlightParkedAtSimTime(flight, t)) return null;
    if (typeof isFlightTimelineStationaryAtSimTime !== 'function' || !isFlightTimelineStationaryAtSimTime(flight, t)) return null;
    const m = flight.timeline_meta;
    if (!m) return null;
    const eibtList = Array.isArray(m.eibtSecList) ? m.eibtSecList : (typeof m.eibtSec === 'number' ? [m.eibtSec] : []);
    const eobtList = Array.isArray(m.eobtSecList) ? m.eobtSecList : (typeof m.eobtSec === 'number' ? [m.eobtSec] : []);
    const nInt = Math.min(eibtList.length, eobtList.length);
    for (let i = 0; i < nInt; i++) {
      const a = Number(eibtList[i]), b = Number(eobtList[i]);
      if (!(isFinite(a) && isFinite(b) && t >= a - 1e-3 && t <= b + 1e-3)) continue;
      const sid = typeof standIdForParkedApronInterval === 'function' ? standIdForParkedApronInterval(flight, t) : '';
      let trTag = '|0|||';
      const cpt = typeof compactPlaybackTrackForFlight === 'function' ? compactPlaybackTrackForFlight(flight) : null;
      if (cpt && Array.isArray(cpt.t) && cpt.t.length) {
        trTag = '|cp|' + cpt.t.length + '|' + cpt.t[0] + '|' + cpt.t[cpt.t.length - 1];
      } else if (flight.timeline && flight.timeline.length) {
        const tl0 = flight.timeline[0], tlZ = flight.timeline[flight.timeline.length - 1];
        trTag = '|tl|' + flight.timeline.length + '|' + tl0.t + '|' + tlZ.t;
      }
      const key = String(flight.id) + '|' + String(sid || '') + '|' + i + '|' + a + '|' + b + trTag;
      return { key: key, anchorT: a };
    }
    return null;
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
