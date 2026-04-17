      const runwayEndBlastInput = document.getElementById('runwayEndBlastPadM');
      if (runwayEndBlastInput && tw.pathType === 'runway') runwayEndBlastInput.value = String(getEffectiveRunwayEndBlastPadM(tw));
      if (maxExitInput) maxExitInput.value = tw.maxExitVelocity != null ? tw.maxExitVelocity : 30;
      if (minExitInput) {
        const minVal = (typeof tw.minExitVelocity === 'number' && isFinite(tw.minExitVelocity) && tw.minExitVelocity > 0)
          ? tw.minExitVelocity
          : 15;
        minExitInput.value = minVal;
      }
      if (tw.pathType === 'runway_exit') {
        const allow = getTaxiwayAllowedRunwayDirections(tw);
        renderRunwayDirectionChoices(allow);
      } else {
        renderRunwayDirectionChoices([]);
      }
      const modeSel = document.getElementById('taxiwayDirectionMode');
      let d = getTaxiwayDirection(tw);
      if (tw.pathType === 'runway' && d === 'both') d = 'clockwise';
      if (modeSel) modeSel.value = d;
      const kindSel = document.getElementById('taxiwayPathTypeKind');
      if (kindSel) {
        const ptk = tw.pathType || 'taxiway';
        kindSel.value = (ptk === 'general_queue_taxiway') ? 'queue' : 'normal';
      }
      syncPathPavementRadiosToValue(pathPavementResolvedForTaxiway(tw));
    } else if (state.selectedObject && state.selectedObject.type === 'apronLink') {
      const lk = state.selectedObject.obj;
      const nameInput = document.getElementById('apronLinkName');
      if (nameInput) nameInput.value = getApronLinkDisplayName(lk);
    } else if (state.selectedObject && state.selectedObject.type === 'layoutEdge') {
      const ed = state.selectedObject.obj;
      const nameInput = document.getElementById('edgeName');
      if (nameInput) nameInput.value = getLayoutEdgeDisplayName(ed);
    } else {
      const rm = settingModeSelect ? settingModeSelect.value : '';
      if (isPathLayoutMode(rm)) {
        const ptx = pathTypeFromLayoutMode(rm);
        syncPathFieldVisibilityForPathType(ptx);
        if (ptx === 'runway_exit') {
          const allowDef = (RW_EXIT_ALLOWED_DEFAULT && RW_EXIT_ALLOWED_DEFAULT.length) ? RW_EXIT_ALLOWED_DEFAULT : ['clockwise', 'counter_clockwise'];
          renderRunwayDirectionChoices(allowDef);
        }
        const twKindIdle = document.getElementById('taxiwayPathTypeKind');
        if (twKindIdle && ptx === 'taxiway') twKindIdle.value = 'normal';
        syncPathPavementRadiosToValue(pathPavementDefaultForPathType(ptx));
      }
      else {
        const maxExitWrap = document.getElementById('runwayMaxExitVelWrap');
        if (maxExitWrap) maxExitWrap.style.display = 'none';
        const minExitWrap = document.getElementById('runwayMinExitVelWrap');
        if (minExitWrap) minExitWrap.style.display = 'none';
        const runwayMinArrWrap = document.getElementById('runwayMinArrVelocityWrap');
        if (runwayMinArrWrap) runwayMinArrWrap.style.display = 'none';
        const runwayLineupWrap = document.getElementById('runwayLineupDistWrap');
        if (runwayLineupWrap) runwayLineupWrap.style.display = 'none';
        const runwayStartDispWrap = document.getElementById('runwayStartDisplacedThresholdWrap');
        if (runwayStartDispWrap) runwayStartDispWrap.style.display = 'none';
        const runwayStartBlastWrap = document.getElementById('runwayStartBlastPadWrap');
        if (runwayStartBlastWrap) runwayStartBlastWrap.style.display = 'none';
        const runwayEndDispWrap = document.getElementById('runwayEndDisplacedThresholdWrap');
        if (runwayEndDispWrap) runwayEndDispWrap.style.display = 'none';
        const runwayEndBlastWrap = document.getElementById('runwayEndBlastPadWrap');
        if (runwayEndBlastWrap) runwayEndBlastWrap.style.display = 'none';
        const taxiwayAvgWrap = document.getElementById('taxiwayAvgVelocityWrap');
        if (taxiwayAvgWrap) taxiwayAvgWrap.style.display = 'none';
        const rwDirWrap = document.getElementById('runwayExitAllowedDirectionWrap');
        if (rwDirWrap) rwDirWrap.style.display = 'none';
        const taxiwayTypeWrapIdle = document.getElementById('taxiwayTypeWrap');
        if (taxiwayTypeWrapIdle) taxiwayTypeWrapIdle.style.display = 'none';
      }
      const selIsTerminal = state.selectedObject && state.selectedObject.type === 'terminal';
      if (!selIsTerminal) {
        const buildingTypeSel = document.getElementById('buildingType');
        if (buildingTypeSel) {
          buildingTypeSel.innerHTML = getBuildingTypeOptionsHtml(BUILDING_TYPE_DEFAULT);
          buildingTypeSel.value = BUILDING_TYPE_DEFAULT;
        }
        const terminalNameInput = document.getElementById('terminalName');
        if (terminalNameInput && rm === 'terminal') terminalNameInput.value = getDefaultBuildingNameForType(BUILDING_TYPE_DEFAULT, null);
      }
      const standModeSel = document.getElementById('standCategoryMode');
      if (standModeSel) standModeSel.value = normalizeStandCategoryMode(_pbbTier.defaultCategoryMode, 'icao');
      syncStandConstraintVisibility('stand', standModeSel ? standModeSel.value : 'icao');
      renderAircraftConstraintChoices('standAircraftAccess', []);
      const remoteModeSel = document.getElementById('remoteCategoryMode');
      if (remoteModeSel) remoteModeSel.value = normalizeStandCategoryMode(_remoteTier.defaultCategoryMode, 'icao');
      syncStandConstraintVisibility('remote', remoteModeSel ? remoteModeSel.value : 'icao');
      renderAircraftConstraintChoices('remoteAircraftAccess', []);
      renderRemoteTerminalAccessChoices([]);
      const tempModeSel = document.getElementById('tempStandCategoryMode');
      if (tempModeSel) tempModeSel.value = normalizeStandCategoryMode(_remoteTier.defaultCategoryMode, 'icao');
      syncStandConstraintVisibility('tempStand', tempModeSel ? tempModeSel.value : 'icao');
      renderAircraftConstraintChoices('tempStandAircraftAccess', []);
      renderTempStandTerminalAccessChoices([]);
      const tempStandNameInput = document.getElementById('tempStandName');
      if (tempStandNameInput && rm === 'tempStand' && !(state.selectedObject && state.selectedObject.type === 'tempStand')) tempStandNameInput.value = '';
      const standNameInputIdle = document.getElementById('standName');
      if (standNameInputIdle && rm === 'pbb' && !(state.selectedObject && state.selectedObject.type === 'pbb')) standNameInputIdle.value = '';
      const remoteNameInputIdle = document.getElementById('remoteName');
      if (remoteNameInputIdle && rm === 'remote' && !(state.selectedObject && state.selectedObject.type === 'remote')) remoteNameInputIdle.value = '';
      const apronLinkNameInput = document.getElementById('apronLinkName');
      if (apronLinkNameInput && rm === 'apronTaxiway') apronLinkNameInput.value = '';
      const edgeNameInput = document.getElementById('edgeName');
      if (edgeNameInput && rm === 'edge') edgeNameInput.value = '';
      const holdingPointNameInput = document.getElementById('holdingPointName');
      if (holdingPointNameInput && rm === 'holdingPoint') holdingPointNameInput.value = getDefaultHoldingPointLabel();
    }
    syncDrawToggleButton('btnTaxiwayDraw', !!state.taxiwayDrawingId);
    syncDrawToggleButton('btnApronLinkDraw', !!state.apronLinkDrawing);
    syncDrawToggleButton('btnPbbDraw', !!state.pbbDrawing);
    syncDrawToggleButton('btnRemoteDraw', !!state.remoteDrawing);
    syncDrawToggleButton('btnTempStandDraw', !!state.tempStandDrawing);
    syncDrawToggleButton('btnHoldingPointDraw', !!state.holdingPointDrawing);
    syncDrawToggleButton('btnMarkerDraw', !!state.markerDrawing);
    renderObjectList();
  }

  function syncStateFromPanel() {
    var el = function(id) { return document.getElementById(id); };
    if (el('gridCellSize')) CELL_SIZE = Math.max(5, Number(el('gridCellSize').value) || 5);
    if (el('gridCols')) GRID_COLS = Math.max(5, Math.min(1000, parseInt(el('gridCols').value, 10) || 200));
    if (el('gridRows')) GRID_ROWS = Math.max(5, Math.min(1000, parseInt(el('gridRows').value, 10) || 200));
    if (state.layoutImageOverlay) {
      state.layoutImageOverlay.opacity = clampLayoutImageOpacity(el('gridLayoutImageOpacity') ? el('gridLayoutImageOpacity').value : state.layoutImageOverlay.opacity);
      state.layoutImageOverlay.widthM = clampLayoutImageSize(el('gridLayoutImageWidthM') ? el('gridLayoutImageWidthM').value : state.layoutImageOverlay.widthM, state.layoutImageOverlay.widthM);
      state.layoutImageOverlay.heightM = clampLayoutImageSize(el('gridLayoutImageHeightM') ? el('gridLayoutImageHeightM').value : state.layoutImageOverlay.heightM, state.layoutImageOverlay.heightM);
      state.layoutImageOverlay.topLeftCol = clampLayoutImagePoint(el('gridLayoutImageCol') ? el('gridLayoutImageCol').value : state.layoutImageOverlay.topLeftCol, state.layoutImageOverlay.topLeftCol);
      state.layoutImageOverlay.topLeftRow = clampLayoutImagePoint(el('gridLayoutImageRow') ? el('gridLayoutImageRow').value : state.layoutImageOverlay.topLeftRow, state.layoutImageOverlay.topLeftRow);
    }
    var t = getCurrentTerminal();
    if (t) {
      if (el('terminalName')) {
        const rawTn = (el('terminalName').value || '').trim();
        if (rawTn && findDuplicateLayoutName('terminal', t.id, rawTn)) {
          alertDuplicateLayoutName();
          el('terminalName').value = t.name || '';
        } else {
          t.name = rawTn || t.name;
        }
      }
      if (el('buildingType')) t.buildingType = normalizeBuildingType(el('buildingType').value || t.buildingType);
      if (el('terminalFloors')) t.floors = Math.max(1, parseInt(el('terminalFloors').value, 10) || 1);
      if (el('terminalFloorToFloor')) t.floorToFloor = Math.max(0.5, Number(el('terminalFloorToFloor').value) || 4);
      t.floorHeight = (t.floors || 1) * (t.floorToFloor || 4);
      if (el('terminalDepartureCapacity')) t.departureCapacity = Math.max(0, parseInt(el('terminalDepartureCapacity').value, 10) || 0);
      if (el('terminalArrivalCapacity')) t.arrivalCapacity = Math.max(0, parseInt(el('terminalArrivalCapacity').value, 10) || 0);
    }
    if (state.selectedObject && state.selectedObject.type === 'pbb') {
      var pbb = state.selectedObject.obj;
      if (el('standName')) {
        const rawSn = (el('standName').value || '').trim();
        if (rawSn && findDuplicateLayoutName('pbb', pbb.id, rawSn)) {
          alertDuplicateLayoutName();
          el('standName').value = pbb.name || '';
        } else {
          pbb.name = rawSn;
        }
      }
      pbb.categoryMode = normalizeStandCategoryMode(el('standCategoryMode') ? el('standCategoryMode').value : pbb.categoryMode, _pbbTier.defaultCategoryMode || 'icao');
      if (el('standCategory')) pbb.category = el('standCategory').value || 'C';
      pbb.allowedAircraftTypes = readCheckedDataItemIds('standAircraftAccess', '.aircraft-type-check');
    }
    if (state.selectedObject && state.selectedObject.type === 'remote') {
      var st = state.selectedObject.obj;
      if (el('remoteName')) {
        const rawRn = (el('remoteName').value || '').trim();
        if (rawRn && findDuplicateLayoutName('remote', st.id, rawRn)) {
          alertDuplicateLayoutName();
          el('remoteName').value = st.name || '';
        } else {
          st.name = rawRn;
        }
      }
      st.categoryMode = normalizeStandCategoryMode(el('remoteCategoryMode') ? el('remoteCategoryMode').value : st.categoryMode, _remoteTier.defaultCategoryMode || 'icao');
      if (el('remoteCategory')) st.category = el('remoteCategory').value || 'C';
      st.allowedAircraftTypes = readCheckedDataItemIds('remoteAircraftAccess', '.aircraft-type-check');
      const accWrap = document.getElementById('remoteTerminalAccess');
      if (accWrap) {
        const checks = accWrap.querySelectorAll('.remote-term-check');
        const allowed = [];
        checks.forEach(function(ch) {
          if (ch.checked) {
            const id = ch.getAttribute('data-item-id');
            if (id) allowed.push(id);
          }
        });
        st.allowedTerminals = allowed;
      }
    }
    if (state.selectedObject && state.selectedObject.type === 'tempStand') {
      var tst = state.selectedObject.obj;
      if (el('tempStandName')) {
        const rawTn = (el('tempStandName').value || '').trim();
        if (rawTn && findDuplicateLayoutName('tempStand', tst.id, rawTn)) {
          alertDuplicateLayoutName();
          el('tempStandName').value = tst.name || '';
        } else {
          tst.name = rawTn;
        }
      }
      tst.categoryMode = normalizeStandCategoryMode(el('tempStandCategoryMode') ? el('tempStandCategoryMode').value : tst.categoryMode, _remoteTier.defaultCategoryMode || 'icao');
      if (el('tempStandCategory')) tst.category = el('tempStandCategory').value || 'C';
      tst.allowedAircraftTypes = readCheckedDataItemIds('tempStandAircraftAccess', '.aircraft-type-check');
      const tempAccWrap = document.getElementById('tempStandTerminalAccess');
      if (tempAccWrap) {
        const checks = tempAccWrap.querySelectorAll('.remote-term-check');
        const allowed = [];
        checks.forEach(function(ch) {
          if (ch.checked) {
            const id = ch.getAttribute('data-item-id');
            if (id) allowed.push(id);
          }
        });
        tst.allowedTerminals = allowed;
      }
    }
    if (state.selectedObject && state.selectedObject.type === 'holdingPoint') {
      var hpo = state.selectedObject.obj;
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
