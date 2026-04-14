      if (el('holdingPointName')) {
        const rawHp = (el('holdingPointName').value || '').trim();
        if (rawHp && findDuplicateLayoutName('holdingPoint', hpo.id, rawHp)) {
          alertDuplicateLayoutName();
          el('holdingPointName').value = hpo.name || '';
        } else {
          hpo.name = rawHp;
        }
      }
    }
    if (state.selectedObject && state.selectedObject.type === 'taxiway') {
      var tw = state.selectedObject.obj;
      if (el('taxiwayName')) {
        const rawTw = (el('taxiwayName').value || '').trim();
        if (rawTw && findDuplicateLayoutName('taxiway', tw.id, rawTw)) {
          alertDuplicateLayoutName();
          el('taxiwayName').value = tw.name || '';
        } else {
          tw.name = rawTw;
        }
      }
      if (el('taxiwayWidth')) {
        const pathType = tw.pathType || 'taxiway';
        const fb = pathType === 'runway' ? RUNWAY_PATH_DEFAULT_WIDTH : (pathType === 'runway_exit' ? RUNWAY_EXIT_DEFAULT_WIDTH : TAXIWAY_DEFAULT_WIDTH);
        tw.width = clampTaxiwayWidthM(pathType, el('taxiwayWidth').value, fb);
      }
      if (document.getElementById('pathPavement')) {
        tw.pavement = getPathPavementFromPanelForPathType(tw.pathType || 'taxiway');
      }
      if (el('taxiwayMaxExitVel')) {
        const mv = Number(el('taxiwayMaxExitVel').value);
        if (tw.pathType === 'runway_exit') tw.maxExitVelocity = isFinite(mv) && mv > 0 ? mv : null;
        else delete tw.maxExitVelocity;
      }
      if (el('taxiwayMinExitVel') && tw.pathType === 'runway_exit') {
        const mv2 = Number(el('taxiwayMinExitVel').value);
        let v = isFinite(mv2) && mv2 > 0 ? mv2 : 15;
        if (typeof tw.maxExitVelocity === 'number' && isFinite(tw.maxExitVelocity) && v > tw.maxExitVelocity) v = tw.maxExitVelocity;
        tw.minExitVelocity = v;
        tw.allowedRwDirections = getRunwayExitAllowedDirectionsFromPanel();
      } else if (tw.pathType !== 'runway_exit') {
        delete tw.minExitVelocity;
        delete tw.allowedRwDirections;
      }
      if (el('taxiwayDirectionMode')) {
        let dirVal = el('taxiwayDirectionMode').value || '';
        if (tw.pathType === 'runway') {
          runwayReverseVerticesIfDirectionChanged(tw, dirVal);
          tw.direction = (dirVal === 'counter_clockwise') ? 'counter_clockwise' : 'clockwise';
        } else tw.direction = dirVal || 'both';
      }
      if (el('taxiwayPathTypeKind')) {
        const ptCur = tw.pathType || 'taxiway';
        if (ptCur === 'taxiway' || ptCur === 'general_queue_taxiway') {
          const kind = String(el('taxiwayPathTypeKind').value || 'normal');
          tw.pathType = (kind === 'queue') ? 'general_queue_taxiway' : 'taxiway';
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
    }
    if (isPathLayoutMode(mode)) {
      const pt = pathTypeFromLayoutMode(mode);
      syncPathFieldVisibilityForPathType(pt);
      if (!state.selectedObject || state.selectedObject.type !== 'taxiway') {
        const nameInput = document.getElementById('taxiwayName');
        if (nameInput) nameInput.value = getDefaultPathName(pt);
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
