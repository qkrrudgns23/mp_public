        }
        lenInput.value = String(Math.max(1, Math.round(arm)));
      }
      if (angleInput) angleInput.value = String(Math.round(getPbbAngleDeg(pbb)));
      if (pbbCountInput) pbbCountInput.value = String(Math.max(1, parseInt(pbb.pbbCount, 10) || 1));
      const boardingWInput = document.getElementById('pbbBoardingWidth');
      const boardingHInput = document.getElementById('pbbBoardingHeight');
      if (boardingWInput) boardingWInput.value = String(getPbbBoardingWidthM(pbb));
      if (boardingHInput) boardingHInput.value = String(getPbbBoardingHeightM(pbb));
      syncStandConstraintVisibility('stand');
      renderAircraftConstraintChoices('standAircraftAccess', getStandAllowedAircraftTypes(pbb), pbb.allowedIcaoCategories);
    }
    if (state.selectedObject && state.selectedObject.type === 'remote') {
      const st = state.selectedObject.obj;
      const nameInput = document.getElementById('remoteName');
      if (nameInput) nameInput.value = st.name || '';
      applyIcaoCategoriesToHost('remoteIcaoCategories', normalizeAllowedIcaoCategories(st.allowedIcaoCategories));
      syncStandConstraintVisibility('remote');
      renderAircraftConstraintChoices('remoteAircraftAccess', getStandAllowedAircraftTypes(st), st.allowedIcaoCategories);
      renderRemoteTerminalAccessChoices(Array.isArray(st.allowedTerminals) ? st.allowedTerminals : []);
    }
    if (state.selectedObject && state.selectedObject.type === 'tempStand') {
      const st = state.selectedObject.obj;
      const nameInput = document.getElementById('tempStandName');
      if (nameInput) nameInput.value = st.name || '';
      applyIcaoCategoriesToHost('tempStandIcaoCategories', normalizeAllowedIcaoCategories(st.allowedIcaoCategories));
      syncStandConstraintVisibility('tempStand');
      renderAircraftConstraintChoices('tempStandAircraftAccess', getStandAllowedAircraftTypes(st), st.allowedIcaoCategories);
      renderTempStandTerminalAccessChoices(Array.isArray(st.allowedTerminals) ? st.allowedTerminals : []);
    }
    if (state.selectedObject && state.selectedObject.type === 'holdingPoint') {
      const hp = state.selectedObject.obj;
      const nameInput = document.getElementById('holdingPointName');
      if (nameInput) nameInput.value = hp.name || '';
    }
    if (state.selectedObject && state.selectedObject.type === 'taxiway') {
      const tw = state.selectedObject.obj;
      const nameInput = document.getElementById('taxiwayName');
      const widthInput = document.getElementById('taxiwayWidth');
      const maxExitInput = document.getElementById('taxiwayMaxExitVel');
      const minExitInput = document.getElementById('taxiwayMinExitVel');
      if (nameInput) nameInput.value = tw.name || '';
      const widthDefault = tw.pathType === 'runway'
        ? RUNWAY_PATH_DEFAULT_WIDTH
        : (tw.pathType === 'runway_exit' ? RUNWAY_EXIT_DEFAULT_WIDTH : TAXIWAY_DEFAULT_WIDTH);
      if (widthInput) widthInput.value = tw.width != null ? tw.width : widthDefault;
      const avgVelInput = document.getElementById('taxiwayAvgMoveVelocity');
      if (avgVelInput) avgVelInput.value = (tw.avgMoveVelocity != null ? tw.avgMoveVelocity : 10);
      syncPathFieldVisibilityForPathType(tw.pathType || 'taxiway');
      const runwayMinArrInput = document.getElementById('runwayMinArrVelocity');
      if (runwayMinArrInput) {
        const mav = (typeof tw.minArrVelocity === 'number' && isFinite(tw.minArrVelocity) && tw.minArrVelocity > 0)
          ? Math.max(1, Math.min(150, tw.minArrVelocity))
          : 15;
        runwayMinArrInput.value = mav;
      }
      const runwayLineupInputCw = document.getElementById('runwayLineupDistM_CW');
      const runwayLineupInputCcw = document.getElementById('runwayLineupDistM_CCW');
      if (tw.pathType === 'runway') {
        if (runwayLineupInputCw) runwayLineupInputCw.value = String(getRunwayLineupDistMByDirection(tw, 'clockwise'));
        if (runwayLineupInputCcw) runwayLineupInputCcw.value = String(getRunwayLineupDistMByDirection(tw, 'counter_clockwise'));
      }
      const runwayStartDispInput = document.getElementById('runwayStartDisplacedThresholdM');
      if (runwayStartDispInput && tw.pathType === 'runway') runwayStartDispInput.value = String(getEffectiveRunwayStartDisplacedThresholdM(tw));
      const runwayStartBlastInput = document.getElementById('runwayStartBlastPadM');
      if (runwayStartBlastInput && tw.pathType === 'runway') runwayStartBlastInput.value = String(getEffectiveRunwayStartBlastPadM(tw));
      const runwayEndDispInput = document.getElementById('runwayEndDisplacedThresholdM');
      if (runwayEndDispInput && tw.pathType === 'runway') runwayEndDispInput.value = String(getEffectiveRunwayEndDisplacedThresholdM(tw));
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
      const skipStandPanelIdleReset = state.selectedObject && (
        state.selectedObject.type === 'pbb' ||
        state.selectedObject.type === 'remote' ||
        state.selectedObject.type === 'tempStand'
      );
      if (!skipStandPanelIdleReset) {
        applyIcaoCategoriesToHost('standIcaoCategories', ['C']);
        syncStandConstraintVisibility('stand');
        renderAircraftConstraintChoices('standAircraftAccess', aircraftTypeIdsForIcaoLetters(['C']), ['C']);
        applyIcaoCategoriesToHost('remoteIcaoCategories', ['C']);
        syncStandConstraintVisibility('remote');
        renderAircraftConstraintChoices('remoteAircraftAccess', aircraftTypeIdsForIcaoLetters(['C']), ['C']);
        renderRemoteTerminalAccessChoices([]);
        applyIcaoCategoriesToHost('tempStandIcaoCategories', ['C']);
        syncStandConstraintVisibility('tempStand');
        renderAircraftConstraintChoices('tempStandAircraftAccess', aircraftTypeIdsForIcaoLetters(['C']), ['C']);
        renderTempStandTerminalAccessChoices([]);
      }
      const tempStandNameInput = document.getElementById('tempStandName');
      if (tempStandNameInput && rm === 'tempStand' && !(state.selectedObject && state.selectedObject.type === 'tempStand')) tempStandNameInput.value = '';
      const standNameInputIdle = document.getElementById('standName');
      if (standNameInputIdle && rm === 'pbb' && !(state.selectedObject && state.selectedObject.type === 'pbb')) standNameInputIdle.value = '';
      const remoteNameInputIdle = document.getElementById('remoteName');
      if (remoteNameInputIdle && rm === 'remote' && !(state.selectedObject && state.selectedObject.type === 'remote')) remoteNameInputIdle.value = '';
      const taxiwayNameInputIdle = document.getElementById('taxiwayName');
      if (taxiwayNameInputIdle && isPathLayoutMode(rm) && !(state.selectedObject && state.selectedObject.type === 'taxiway')) taxiwayNameInputIdle.value = '';
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
      applyUnifiedStandConstraintFromPanelToObject(pbb, 'standIcaoCategories', 'standAircraftAccess');
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
      applyUnifiedStandConstraintFromPanelToObject(st, 'remoteIcaoCategories', 'remoteAircraftAccess');
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
      applyUnifiedStandConstraintFromPanelToObject(tst, 'tempStandIcaoCategories', 'tempStandAircraftAccess');
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
